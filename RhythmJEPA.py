import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.models.layers import DropPath
except ImportError:
    from timm.layers import DropPath
from mamba_ssm import Mamba


def to_bcthw(x):
    if x.dim() != 5:
        raise ValueError(f"Expected 5D video, got {tuple(x.shape)}")
    if x.shape[1] == 3:
        return x
    if x.shape[2] == 3:
        return x.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"Expected [B,3,T,H,W] or [B,T,3,H,W], got {tuple(x.shape)}")


def norm_wave(x, eps=1e-6):
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(eps)


def conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


def conv_bn_gelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False), nn.BatchNorm2d(out_channels), nn.GELU())


def stem_layer(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=180, max_h=16, max_w=16):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_len, self.max_h, self.max_w = max_len, max_h, max_w
        self.time = nn.Embedding(max_len, dim)
        self.height = nn.Embedding(max_h, dim)
        self.width = nn.Embedding(max_w, dim)

    def temporal_tokens(self, length, device):
        if length > self.max_len:
            raise ValueError(f"Temporal length {length} exceeds max_len {self.max_len}")
        return self.time(torch.arange(length, device=device)).unsqueeze(0) * self.scale

    def forward(self, x):
        _, c, t, h, w = x.shape
        if t > self.max_len or h > self.max_h or w > self.max_w:
            raise ValueError(f"Feature size {(t, h, w)} exceeds max size {(self.max_len, self.max_h, self.max_w)}")
        pt = self.temporal_tokens(t, x.device).transpose(1, 2).view(1, c, t, 1, 1)
        ph = self.height(torch.arange(h, device=x.device)).transpose(0, 1).view(1, c, 1, h, 1) * self.scale
        pw = self.width(torch.arange(w, device=x.device)).transpose(0, 1).view(1, c, 1, 1, w) * self.scale
        return x + pt + ph + pw


class FacialFeatureStem(nn.Module):
    def __init__(self, in_channels=3, stem_channels=64):
        super().__init__()
        self.original_stem = stem_layer(in_channels, stem_channels, 5, 2, 2)
        self.difference_stem = stem_layer(in_channels * 4, stem_channels, 5, 2, 2)
        self.fusion_stem = stem_layer(stem_channels, stem_channels, 3, 1, 1)
        self.motion_stem = stem_layer(stem_channels, stem_channels, 3, 1, 1)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.beta1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.beta2 = nn.Parameter(torch.tensor(0.5))

    def apply_framewise(self, layer, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        x = layer(x)
        _, c, h, w = x.shape
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def temporal_inputs(self, x):
        b, c, t, h, w = x.shape
        x = F.pad(x, (0, 0, 0, 0, 2, 2), mode="replicate")
        x_t_2, x_t_1, x_t, x_t1, x_t2 = x[:, :, 0:t], x[:, :, 1:t + 1], x[:, :, 2:t + 2], x[:, :, 3:t + 3], x[:, :, 4:t + 4]
        d_t_1, d_t, d_t1, d_t2 = x_t_1 - x_t_2, x_t - x_t_1, x_t1 - x_t, x_t2 - x_t1
        return x_t, torch.cat([d_t_1, d_t, d_t1, d_t2], dim=1)

    def forward(self, x):
        x_original, x_difference = self.temporal_inputs(x)
        x_original = self.apply_framewise(self.original_stem, x_original)
        x_difference = self.apply_framewise(self.difference_stem, x_difference)
        x_fusion = self.apply_framewise(self.fusion_stem, self.alpha1 * x_original + self.beta1 * x_difference)
        x_motion = self.apply_framewise(self.motion_stem, x_difference)
        return self.alpha2 * x_fusion + self.beta2 * x_motion


class AxialLocalFacialMixer(nn.Module):
    def __init__(self, dim, gamma_init=1e-2):
        super().__init__()
        self.horizontal_mixer = conv_bn_gelu(dim, dim, (1, 5), 1, (0, 2), groups=dim)
        self.vertical_mixer = conv_bn_gelu(dim, dim, (5, 1), 1, (2, 0), groups=dim)
        self.local_mixer = conv_bn_gelu(dim, dim, 3, 1, 1, groups=dim)
        self.fusion = nn.Sequential(nn.Conv2d(dim * 3, dim, 1, bias=False), nn.BatchNorm2d(dim))
        self.gamma = nn.Parameter(torch.ones(1) * gamma_init)

    def forward(self, x):
        dx = torch.cat([self.horizontal_mixer(x), self.vertical_mixer(x), self.local_mixer(x)], dim=1)
        return x + self.gamma * self.fusion(dx)


class AttentiveSpatialPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Conv2d(dim, 1, 1)

    def forward(self, x):
        a = self.score(x).flatten(2).softmax(dim=-1)
        return torch.sum(x.flatten(2) * a, dim=-1)


class FacialTokenEncoder(nn.Module):
    def __init__(self, in_channels=3, stem_channels=64, token_dim=96, max_len=180, max_h=16, max_w=16):
        super().__init__()
        self.stem = FacialFeatureStem(in_channels, stem_channels)
        self.proj = conv_bn_relu(stem_channels, token_dim, 1)
        self.position = PositionalEmbedding(token_dim, max_len, max_h, max_w)
        self.spatial = AxialLocalFacialMixer(token_dim)
        self.pool = AttentiveSpatialPooling(token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def apply_framewise(self, layer, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        x = layer(x)
        _, c, h, w = x.shape
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x):
        x = to_bcthw(x)
        x = self.stem(x)
        x = self.position(self.apply_framewise(self.proj, x))
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        x = self.spatial(x)
        x = self.pool(x).view(b, t, -1)
        return self.norm(x)


class DepthwiseTemporalMixer(nn.Module):
    def __init__(self, dim=192, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.pw = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.scale = nn.Parameter(torch.ones(1) * 1e-2)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h = self.dwconv(h).transpose(1, 2)
        return x + self.scale * self.pw(h)


class FANStateHead(nn.Module):
    def __init__(self, dim=192, num_states=4, hidden_dim=None, periodic_dim=None, nonperiodic_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        periodic_dim = periodic_dim or max(dim // 4, 16)
        nonperiodic_dim = nonperiodic_dim or max(dim // 2, 16)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.periodic = nn.Linear(hidden_dim, periodic_dim)
        self.nonperiodic = nn.Sequential(nn.Linear(hidden_dim, nonperiodic_dim), nn.GELU())
        self.out = nn.Linear(periodic_dim * 2 + nonperiodic_dim, num_states)

    def forward(self, x):
        h = self.in_proj(self.norm(x))
        p = self.periodic(h)
        p = torch.cat([torch.sin(p), torch.cos(p)], dim=-1)
        r = self.nonperiodic(h)
        return self.out(torch.cat([p, r], dim=-1))


class RhythmStatePlanner(nn.Module):
    def __init__(self, dim=192, num_states=4, kernel_size=5, dropout=0.0):
        super().__init__()
        self.mixer = DepthwiseTemporalMixer(dim, kernel_size, dropout)
        self.head = FANStateHead(dim, num_states)

    def forward(self, x, detach_input=False):
        if detach_input:
            x = x.detach()
        return self.head(self.mixer(x))


class BiMambaBlock(nn.Module):
    """Pre-norm bidirectional Mamba block for [B, T, C] token sequences."""

    def __init__(self, dim=192, d_state=16, d_conv=4, expand=2, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(
                f"Expected tokens [B,T,{self.dim}], got {tuple(x.shape)}"
            )
        return self.norm2(x + self.drop_path(self.mamba(self.norm1(x))))


class BiMambaStack(nn.Module):
    def __init__(self, dim=192, depth=3, d_state=16, d_conv=4, expand=2, drop_path=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            BiMambaBlock(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=drop_path,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.blocks(x))


class DualOrderMambaEncoder(nn.Module):
    def __init__(self, dim=192, depth=3, d_state=16, d_conv=4, expand=2, drop_path=0.0):
        super().__init__()
        scan_kwargs = dict(
            dim=dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=drop_path,
        )
        self.time_scan = BiMambaStack(**scan_kwargs)
        self.state_scan = BiMambaStack(**scan_kwargs)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, state_ids):
        b, t, d = x.shape
        if state_ids.shape != (b, t):
            raise ValueError(
                f"Expected state_ids {(b, t)}, got {tuple(state_ids.shape)}"
            )

        y_time = self.time_scan(x)

        time = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        order = (state_ids * t + time).argsort(dim=1)
        gather_index = order.unsqueeze(-1).expand(-1, -1, d)
        x_state = x.gather(1, gather_index)

        y_state = self.state_scan(x_state)
        inverse_order = order.argsort(dim=1)
        scatter_index = inverse_order.unsqueeze(-1).expand(-1, -1, d)
        y_state = y_state.gather(1, scatter_index)

        gate = self.gate(torch.cat([x, y_time, y_state], dim=-1))
        return self.norm(gate * y_time + (1.0 - gate) * y_state)


@torch.no_grad()
def grammar_decode(q, transition):
    b, t, k = q.shape
    logp = q.clamp_min(1e-6).log()
    trans = transition.to(device=q.device).bool()
    neg = torch.finfo(logp.dtype).min / 4
    score = logp[:, 0]
    ptrs = []
    for step in range(1, t):
        cand = score.unsqueeze(2).expand(b, k, k).masked_fill(~trans.unsqueeze(0), neg)
        best, ptr = cand.max(dim=1)
        score = best + logp[:, step]
        ptrs.append(ptr)
    last = score.argmax(dim=1)
    path = torch.empty(b, t, dtype=torch.long, device=q.device)
    path[:, -1] = last
    for step in range(t - 2, -1, -1):
        last = ptrs[step].gather(1, last[:, None]).squeeze(1)
        path[:, step] = last
    return path


class CyRhythmJEPA(nn.Module):
    def __init__(
        self,
        token_encoder=None,
        token_dim=96,
        max_len=180,
        num_states=4,
        depth=3,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        drop_path=0.0,
        ema=0.996,
        mode="pretrain",
        mask_ratio=0.7,
        planner_kernel_size=5,
        detach_state_input=False,
    ):
        super().__init__()
        self.mode = mode
        self.ema = ema
        self.num_states = num_states
        self.mask_ratio = mask_ratio
        self.detach_state_input = detach_state_input
        self.token_encoder = token_encoder if token_encoder is not None else FacialTokenEncoder(token_dim=token_dim, max_len=max_len)
        self.state_planner = RhythmStatePlanner(token_dim, num_states, planner_kernel_size, dropout)
        self.context_encoder = DualOrderMambaEncoder(token_dim, depth, d_state, d_conv, expand, drop_path)
        self.teacher_token_encoder = copy.deepcopy(self.token_encoder)
        self.teacher_state_planner = copy.deepcopy(self.state_planner)
        self.teacher_context_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim * 2), nn.GELU(), nn.Linear(token_dim * 2, token_dim))
        self.rppg_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        self.raw_mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1, 1))
        self.register_buffer("transition", torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=torch.float32))
        self.freeze_teacher()

    def train(self, mode=True):
        super().train(mode)
        self.teacher_token_encoder.eval()
        self.teacher_state_planner.eval()
        self.teacher_context_encoder.eval()
        return self

    def freeze_teacher(self):
        for m in (self.teacher_token_encoder, self.teacher_state_planner, self.teacher_context_encoder):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for mt, ms in ((self.teacher_token_encoder, self.token_encoder), (self.teacher_state_planner, self.state_planner), (self.teacher_context_encoder, self.context_encoder)):
            for pt, ps in zip(mt.parameters(), ms.parameters()):
                pt.mul_(self.ema).add_(ps, alpha=1.0 - self.ema)
            for bt, bs in zip(mt.buffers(), ms.buffers()):
                bt.copy_(bs)

    def sample_random_mask(self, batch, length, device):
        mask = torch.rand(batch, length, device=device) < self.mask_ratio
        all_mask, no_mask = mask.all(dim=1), ~mask.any(dim=1)
        if all_mask.any():
            ids = all_mask.nonzero(as_tuple=False).squeeze(1)
            mask[ids, torch.randint(0, length, (ids.numel(),), device=device)] = False
        if no_mask.any():
            ids = no_mask.nonzero(as_tuple=False).squeeze(1)
            mask[ids, torch.randint(0, length, (ids.numel(),), device=device)] = True
        return mask.detach()

    def mask_video(self, x, mask):
        x = to_bcthw(x)
        token = self.raw_mask_token.sigmoid().expand(x.size(0), -1, x.size(2), x.size(3), x.size(4))
        return torch.where(mask[:, None, :, None, None], token, x)

    def state_probs(self, x, teacher=False, detach_input=False):
        planner = self.teacher_state_planner if teacher else self.state_planner
        return F.softmax(planner(x, detach_input=detach_input), dim=-1)

    def predict_wave(self, x):
        return norm_wave(self.rppg_head(x).squeeze(-1))

    def forward(self, x, return_aux=False):
        x = to_bcthw(x)
        b, _, t, _, _ = x.shape
        if self.mode != "pretrain":
            z = self.token_encoder(x)
            q = self.state_probs(z, detach_input=False)
            state_ids = grammar_decode(q, self.transition)
            r = self.context_encoder(z, state_ids)
            y = self.predict_wave(r)
            if return_aux:
                return y, {"latent": r, "state_q": q, "state_ids": state_ids.detach()}
            return y

        mask = self.sample_random_mask(b, t, x.device)
        z = self.token_encoder(self.mask_video(x, mask))
        q = self.state_probs(z, detach_input=self.detach_state_input)
        state_ids = grammar_decode(q, self.transition)
        r = self.context_encoder(z, state_ids)
        pred = self.predictor(r)
        with torch.no_grad():
            z_t = self.teacher_token_encoder(x)
            q_t = self.state_probs(z_t, teacher=True)
            state_ids_t = grammar_decode(q_t, self.transition)
            target = self.teacher_context_encoder(z_t, state_ids_t)
        y = self.predict_wave(r)
        if not return_aux:
            return y
        return y, {"latent": r, "jepa_pred": pred, "jepa_target": target, "jepa_mask": mask, "state_q": q, "teacher_state_q": q_t, "transition": self.transition, "state_ids": state_ids.detach(), "teacher_state_ids": state_ids_t.detach()}

    @torch.no_grad()
    def predict(self, x):
        mode = self.mode
        self.mode = "finetune"
        y = self.forward(x)
        self.mode = mode
        return y


class CyRhythmRPPG(nn.Module):
    def __init__(self, pretrained, token_dim=96):
        super().__init__()
        self.token_encoder = pretrained.token_encoder
        self.state_planner = pretrained.state_planner
        self.context_encoder = pretrained.context_encoder
        self.register_buffer("transition", pretrained.transition.detach().clone())
        self.head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))

    def forward(self, x):
        z = self.token_encoder(x)
        q = F.softmax(self.state_planner(z), dim=-1)
        state_ids = grammar_decode(q, self.transition)
        return norm_wave(self.head(self.context_encoder(z, state_ids)).squeeze(-1))

    @torch.no_grad()
    def predict(self, x):
        return self(x)


__all__ = ["CyRhythmJEPA", "CyRhythmRPPG", "FacialTokenEncoder", "RhythmStatePlanner", "DualOrderMambaEncoder", "BiMambaBlock", "BiMambaStack", "FANStateHead", "DepthwiseTemporalMixer", "AxialLocalFacialMixer"]
