# Vision-RWKV-7: RWKV-7 vision backbone with bidirectional scanning, Q-Shift,
#                gated fusion, input-dependent mixing, layer scale,
#                extra normalization, and multi-scale feature output.
#
# Features 1-11: Q-Shift, Bidirectional Scan, Gated Fusion,
# Flexible Decay, Bounded Exps, Extra LN, Layer Scale, etc.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Sequence, List

HEAD_SIZE = 64
TIME_MIX_EXTRA_DIM = 32

# =====================================================================
# Utility modules
# =====================================================================


class Permute(nn.Module):
    """Channel-permute layer (compat for nn.Permute)."""

    __slots__ = ("dims",)

    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


# =====================================================================
# Helpers
# =====================================================================


def q_shift_multihead(
    input,
    shift_pixel=1,
    head_dim=HEAD_SIZE,
    patch_resolution: Tuple[int, int] = (14, 14),
    with_cls_token=False,
):
    """Q-Shift: 4-directional 2D token shift along channel groups.
    Feature 1 (Vision-RWKV) + Feature 11 (2D local pre-mixer).
    Ported from VRWKV6 / vrwkv6.py:75-100.
    """
    B, N, C = input.shape
    assert C % head_dim == 0, f"C={C} not divisible by head_dim={head_dim}"
    assert head_dim % 4 == 0, f"head_dim={head_dim} not divisible by 4"
    n_head = C // head_dim

    cls_tokens = input[:, [-1], :]
    if with_cls_token:
        input = input[:, :-1, :]
        N = N - 1

    # [B, n_head, head_dim, H, W]
    input = input.transpose(1, 2).reshape(
        B, n_head, head_dim, patch_resolution[0], patch_resolution[1]
    )
    _, _, _, H, W = input.shape

    output = torch.zeros_like(input)
    # Group 0: shift right by 1
    if shift_pixel < W:
        output[:, :, 0 : int(head_dim * 1 / 4), :, shift_pixel:W] = input[
            :, :, 0 : int(head_dim * 1 / 4), :, 0 : W - shift_pixel
        ]
    # Group 1: shift left by 1
    if shift_pixel < W:
        output[:, :, int(head_dim / 4) : int(head_dim / 2), :, 0 : W - shift_pixel] = (
            input[:, :, int(head_dim / 4) : int(head_dim / 2), :, shift_pixel:W]
        )
    # Group 2: shift down by 1
    if shift_pixel < H:
        output[:, :, int(head_dim / 2) : int(head_dim / 4 * 3), shift_pixel:H, :] = (
            input[
                :, :, int(head_dim / 2) : int(head_dim / 4 * 3), 0 : H - shift_pixel, :
            ]
        )
    # Group 3: shift up by 1
    if shift_pixel < H:
        output[:, :, int(head_dim * 3 / 4) : int(head_dim), 0 : H - shift_pixel, :] = (
            input[:, :, int(head_dim * 3 / 4) : int(head_dim), shift_pixel:H, :]
        )

    output = output.reshape(B, C, N).transpose(1, 2)
    if with_cls_token:
        output = torch.cat((output, cls_tokens), dim=1)
    return output


def resize_pos_embed(
    pos_embed, src_shape, dst_shape, mode="bicubic", num_extra_tokens=0
):
    """Interpolate 2D positional embedding to different resolution.
    Ported from AudioRWKV vrwkv6.py:626-667.
    """
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    if src_h == dst_h and src_w == dst_w:
        return pos_embed
    if num_extra_tokens:
        pos_embed = pos_embed[:, num_extra_tokens:]
    pos_embed = pos_embed.reshape(1, src_h, src_w, -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, size=(dst_h, dst_w), mode=mode, align_corners=False
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[1])
    return pos_embed


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    __slots__ = ("drop_prob",)

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# =====================================================================
# Vision_RWKV7_Block
# =====================================================================


class Vision_RWKV7_Block(nn.Module):
    """Vision-RWKV-7 block.

    Architecture:
      ln0(layer0) -> ln1 -> QShift -> input-dependent spatial offsets precompute
        -> forward scan (1D shift + QShift combined, RWKV-7 delta rule)
        -> backward scan
        -> gated fusion -> gamma1 -> att_ln -> +residual
      -> ln2 -> QShift -> ReLU^2 MLP -> gamma2 -> ffn_ln -> +residual
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_layer: int,
        layer_id: int,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        with_cls_token: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
        assert self.head_size * n_head == n_embd
        self.with_cls_token = with_cls_token

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        # ================================================================
        # RWKV-7  HEAD-VARIANT PARAMETERS
        # ================================================================
        # -- 1D scan-path token shift (core RWKV-7: sx * self.x[h]) --
        #   [6, D]:  r, w, k, v, a, g  — per-channel lerp with prev token
        self.x = nn.Parameter(torch.zeros(6, n_embd))

        # -- Q-Shift residual mixing  (static per-head, per-channel) --
        self.time_maa_x = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_w = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_v = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_r = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_g = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_a = nn.Parameter(torch.zeros(1, 1, n_embd))

        # -- Input-dependent Q-Shift offset via low-rank MLP --
        #   w1: [D, 32*6]  →  w2: [6, 32, D]  →  6 dynamic offsets m{w,k,v,r,g,a}
        self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, TIME_MIX_EXTRA_DIM * 6))
        self.time_maa_w2 = nn.Parameter(torch.zeros(6, TIME_MIX_EXTRA_DIM, n_embd))

        # ================================================================
        # RWKV-7  DELTA-RULE PARAMETERS
        # ================================================================
        # Decay:  w_raw = w0 + tanh(xw @ w1) @ w2
        self.w0 = nn.Parameter(torch.zeros(n_embd))
        self.w1 = nn.Parameter(torch.zeros(n_embd, 32))
        self.w2 = nn.Parameter(torch.zeros(32, n_embd))

        # ICLR:  a = sigmoid(a0 + (x_a @ a1) @ a2)
        self.a0 = nn.Parameter(torch.zeros(n_embd))
        self.a1 = nn.Parameter(torch.zeros(n_embd, 32))
        self.a2 = nn.Parameter(torch.zeros(32, n_embd))

        # Value residual:  nu = sigmoid(v0 + (xv @ v1) @ v2)
        if layer_id != 0:
            self.v0 = nn.Parameter(torch.zeros(n_embd))
        self.v1 = nn.Parameter(torch.zeros(n_embd, 32))
        self.v2 = nn.Parameter(torch.zeros(32, n_embd))

        # Output gate:  g = sigmoid(x_g @ g1) @ g2
        self.g1 = nn.Parameter(torch.zeros(n_embd, 32))
        self.g2 = nn.Parameter(torch.zeros(32, n_embd))

        # Removal key (xi)  and  replacement-rate (alpha)
        self.k_k = nn.Parameter(torch.zeros(n_embd))
        self.k_a = nn.Parameter(torch.zeros(n_embd))

        # Bonus (rho):  [n_head, head_size]
        self.r_k = nn.Parameter(torch.zeros(n_head, self.head_size))

        # Linear projections (no bias)
        self.att_receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.att_key = nn.Linear(n_embd, n_embd, bias=False)
        self.att_value = nn.Linear(n_embd, n_embd, bias=False)
        self.att_output = nn.Linear(n_embd, n_embd, bias=False)

        # Per-head group norm  (RWKV-7 uses eps=64e-5)
        self.att_group_norm = nn.GroupNorm(
            n_head, n_embd, eps=self.n_head * 1e-5, affine=True
        )

        # ================================================================
        # Vision-specific additions
        # ================================================================
        self.fusion_gate = nn.Linear(n_embd, n_embd, bias=False)  # Feat 3
        self.att_ln = nn.LayerNorm(n_embd)  # Feat 7
        self.ffn_ln = nn.LayerNorm(n_embd)  # Feat 7

        if init_values is not None:
            self.gamma1 = nn.Parameter(init_values * torch.ones(n_embd))
            self.gamma2 = nn.Parameter(init_values * torch.ones(n_embd))
        else:
            self.gamma1 = None
            self.gamma2 = None

        # -- Channel-mix (ReLU^2, 4x expansion) --
        self.ffn_x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        dim_ffn = 4 * n_embd
        self.ffn_key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.ffn_value = nn.Linear(dim_ffn, n_embd, bias=False)

        self._init_weights()

    # -----------------------------------------------------------------
    # Weight initialization
    # -----------------------------------------------------------------
    def _init_weights(self):
        with torch.no_grad():
            if self.n_layer <= 1:
                ratio_0_to_1 = 0.0
                ratio_1_to_almost0 = 0.5
            else:
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)

            # Channel index in [0, 1)
            idx = torch.arange(self.n_embd, dtype=torch.float) / max(self.n_embd - 1, 1)
            ddd = idx.view(1, 1, self.n_embd)

            # RWKV-7 1D token-shift params  (small init, will learn)
            self.x.uniform_(-0.01, 0.01)

            # VRWKV6-style fancy time_maa init
            def fancy_mix(base_pow):
                return 1.0 - torch.pow(ddd, base_pow)

            self.time_maa_x.copy_(fancy_mix(ratio_1_to_almost0))
            self.time_maa_w.copy_(fancy_mix(ratio_1_to_almost0))
            self.time_maa_k.copy_(fancy_mix(ratio_1_to_almost0))
            self.time_maa_v.copy_(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r.copy_(fancy_mix(0.5 * ratio_1_to_almost0))
            self.time_maa_g.copy_(fancy_mix(0.5 * ratio_1_to_almost0))
            self.time_maa_a.copy_(fancy_mix(0.5 * ratio_1_to_almost0))

            self.time_maa_w1.uniform_(-1e-4, 1e-4)
            self.time_maa_w2.uniform_(-1e-4, 1e-4)

            # Decay init (Feature 5: flexible range = [-3, ~2])
            decay_speed = -3 + 5 * idx ** (0.7 + 1.3 * ratio_0_to_1)
            self.w0.copy_(decay_speed)

            # Bonus init  (time_faaaa zigzag)
            tmp = torch.zeros(self.n_head, self.head_size)
            for h in range(self.n_head):
                for n in range(self.head_size):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[h, n] = ratio_0_to_1 * (1 - n / (self.head_size - 1)) + zigzag
            self.r_k.copy_(tmp)

            for p in [
                self.w1,
                self.w2,
                self.a1,
                self.a2,
                self.v1,
                self.v2,
                self.g1,
                self.g2,
            ]:
                p.uniform_(-1e-4, 1e-4)

    # -----------------------------------------------------------------
    # Pre-compute spatial residual and input-dependent offsets
    # -----------------------------------------------------------------
    def _spatial_prep(self, xn: torch.Tensor, patch_resolution: Tuple[int, int]):
        """Q-Shift + input-dependent dynamic offsets  (position-only).

        Returns dict with:
          xx:   [B, N, D]  spatial residual
          dm:   [6, B, N, D]  dynamic mixing offsets (state-independent)
        """
        B, N, D = xn.shape

        # Q-Shift: 2D spatial residual (Features 1, 11)
        xs = q_shift_multihead(
            xn,
            shift_pixel=1,
            head_dim=self.head_size,
            patch_resolution=patch_resolution,
            with_cls_token=self.with_cls_token,
        )
        xx = xs - xn  # [B, N, D]

        # Input-dependent dynamic offsets (VRWKV6 jit_func)
        #   base:   x + xx * time_maa_x
        #   MLP:    tanh(base @ w1)  ->  split 6 ways  ->  bmm(w2)
        x_base = xn + xx * self.time_maa_x  # [B, N, D]
        x_dyn = torch.tanh(x_base @ self.time_maa_w1)  # [B, N, 192]
        x_dyn = x_dyn.view(B * N, 6, -1).transpose(0, 1)  # [6, B*N, 32]
        x_dyn = torch.bmm(x_dyn, self.time_maa_w2)  # [6, B*N, D]
        dm = x_dyn.view(6, B, N, D)  # [6, B, N, D]

        return {"xx": xx, "dm": dm}

    # -----------------------------------------------------------------
    # RWKV-7 scan over one direction  (both shifts combined)
    # -----------------------------------------------------------------
    def _scan(
        self,
        xn: torch.Tensor,
        sp: dict,
        direction: str,
        v_first_seq: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """RWKV-7 delta-rule recurrence over the 1D sequence.

        For each token combines:
          - 1D scan shift `sx * self.x[h]`   (original RWKV-7)
          - Q-Shift residual `xx * (static + dynamic)`  (vision adaptation)

        Args:
          xn:   [B, N, D]   pre-normed input.
          sp:   spatial prep dict {xx, dm}.
          direction: 'forward' | 'backward'.
          v_first_seq: [B, N, D] values from layer 0 (for Value Residual).

        Returns: (out [B, N, D], v_first_seq [B, N, D]).
        """
        B, N, D = xn.shape
        Hd = self.n_head
        S = self.head_size
        dev = xn.device

        rev = direction == "backward"

        # Explicitly flip the correct sequence dimension for each tensor
        xn_seq = torch.flip(xn, dims=[1]) if rev else xn
        xx_seq = torch.flip(sp["xx"], dims=[1]) if rev else sp["xx"]

        # sp["dm"] is [6, B, N, D], so the sequence dim is 2
        dm_seq = torch.flip(sp["dm"], dims=[2]) if rev else sp["dm"]

        vf_seq = None
        if v_first_seq is not None:
            # v_first_seq is [B, N, D], so sequence dim is 1
            vf_seq = torch.flip(v_first_seq, dims=[1]) if rev else v_first_seq

        # Fresh state per direction
        state = torch.zeros(B, Hd, S, S, device=dev)
        state_time = torch.zeros(B, D, device=dev)

        # Pre-extract static mixing params for fast access  [D] each
        sw = self.time_maa_w.reshape(-1)
        sk = self.time_maa_k.reshape(-1)
        sv = self.time_maa_v.reshape(-1)
        sr = self.time_maa_r.reshape(-1)
        sg = self.time_maa_g.reshape(-1)
        sa = self.time_maa_a.reshape(-1)
        x0, x1, x2, x3, x4, x5 = self.x.unbind(dim=0)  # [D] each

        outputs = []
        v_first_list = []
        for t in range(N):
            token = xn_seq[:, t, :]  # [B, D]
            xx_t = xx_seq[:, t, :]  # [B, D]
            dm_t = dm_seq[:, :, t, :]  # [6, B, D]
            dmw, dmk, dmv, dmr, dmg, dma = dm_t.unbind(dim=0)

            # ---- 1D scan shift (core RWKV-7) ----
            sx = state_time - token
            state_time.copy_(token)

            # ---- 6 heads: 1D shift + 2D Q-Shift residual ----
            xw = token + sx * x0 + xx_t * (sw + dmw)
            xk = token + sx * x1 + xx_t * (sk + dmk)
            xv = token + sx * x2 + xx_t * (sv + dmv)
            xr = token + sx * x3 + xx_t * (sr + dmr)
            xg_in = token + sx * x4 + xx_t * (sg + dmg)
            xa_in = token + sx * x5 + xx_t * (sa + dma)

            # ---- RWKV-7 delta rule ----

            # Decay
            w_raw = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
            w = torch.exp(-0.606531 * torch.sigmoid(w_raw.float()))

            # Projections
            r = self.att_receptance(xr)
            k = self.att_key(xk)
            v = self.att_value(xv)

            # Value residual (Feature 9: lerp(v_0, v_l, nu))
            if self.layer_id == 0:
                vf = v
                v_first_list.append(vf)
            else:
                vf = vf_seq[:, t, :] if vf_seq is not None else v
                vr = self.v0 + (xv @ self.v1) @ self.v2
                # Paper: lerp(v_0, v_l, nu) = v_0 + (v_l - v_0) * nu
                v = vf + (v - vf) * torch.sigmoid(vr)

            # ICLR and output gate
            a = torch.sigmoid(self.a0 + (xa_in @ self.a1) @ self.a2)
            g = torch.sigmoid(xg_in @ self.g1) @ self.g2

            # Removal key (kappa_hat)
            kk = k * self.k_k
            kk = F.normalize(kk.view(B, Hd, S), dim=-1, p=2.0).view(B, -1)

            # Replacement key (k_tilde)
            kt = k * (1 + (a - 1) * self.k_a)

            # ---- State update (generalized delta rule) ----
            vk = v.view(B, Hd, S, 1) @ kt.view(B, Hd, 1, S)
            ab = (-kk).view(B, Hd, S, 1) @ (kk * a).view(B, Hd, 1, S)
            state = state * w.view(B, Hd, 1, S) + state @ ab.float() + vk.float()

            # ---- Query ----
            r_h = r.view(B, Hd, S).unsqueeze(-1)  # [B, H, S, 1]
            out = (state @ r_h).squeeze(-1)  # [B, H, S]
            out = self.att_group_norm(out.flatten(start_dim=1))

            # Bonus term (Must use kt, the replacement key, per RWKV-7 Eq. 20)
            bonus = (
                (r.view(B, Hd, S) * kt.view(B, Hd, S) * self.r_k.view(Hd, S)).sum(
                    dim=-1, keepdim=True
                )
                * v.view(B, Hd, S)
            ).view(B, D)

            out = (out + bonus) * g
            out = self.att_output(out)
            outputs.append(out)

        out = torch.stack(outputs, dim=1)  # [B, N, D]
        if rev:
            out = torch.flip(out, dims=[1])

        v_first_out = None
        if self.layer_id == 0:
            v_first_out = torch.stack(v_first_list, dim=1)
            if rev:
                v_first_out = torch.flip(v_first_out, dims=[1])

        return out, v_first_out

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        patch_resolution: Tuple[int, int],
        v_first_fwd: Optional[torch.Tensor] = None,
        v_first_bwd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.layer_id == 0:
            x = self.ln0(x)

        # === Time-mix ===
        xn = self.ln1(x)

        # Pre-compute spatial residual + dynamic offsets (position-only)
        sp = self._spatial_prep(xn, patch_resolution)

        # Gate features from local residual (Feature 10)
        x_gate = xn + sp["xx"] * 0.5

        # Bidirectional scans with independent v_first (Features 2, 9)
        out_fwd, vf_fwd = self._scan(xn, sp, "forward", v_first_fwd)
        out_bwd, vf_bwd = self._scan(xn, sp, "backward", v_first_bwd)

        # Gated fusion (Feature 3)
        gate = torch.sigmoid(self.fusion_gate(x_gate))
        att_out = gate * out_fwd + (1 - gate) * out_bwd

        # Layer scale (Feature 8) + extra LN (Feature 7)
        if self.gamma1 is not None:
            att_out = self.gamma1 * att_out
        x = x + self.drop_path(self.att_ln(att_out))

        # === Channel-mix: Q-Shift + ReLU^2 MLP ===
        xn = self.ln2(x)
        xs = q_shift_multihead(
            xn,
            shift_pixel=1,
            head_dim=self.head_size,
            patch_resolution=patch_resolution,
            with_cls_token=self.with_cls_token,
        )
        xx = xs - xn
        xk = xn + xx * self.ffn_x_k

        k = F.relu(self.ffn_key(xk)).pow(2)
        ffn_out = self.ffn_value(k)

        if self.gamma2 is not None:
            ffn_out = self.gamma2 * ffn_out
        x = x + self.drop_path(self.ffn_ln(ffn_out))

        return x, vf_fwd, vf_bwd


# =====================================================================
# Vision_RWKV7   (Backbone)
# =====================================================================


class Vision_RWKV7(nn.Module):
    """Vision-RWKV-7 backbone for replacing ViT.

    Architecture:
        PatchEmbed -> +PosEmbed -> [Vision_RWKV7_Block x depth] -> LN -> multi-scale

    Default (tiny):  embed_dims=192, n_head=3, depth=12  (~20M params)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dims: int = 192,
        num_heads: int = 3,
        depth: int = 12,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 1e-5,
        final_norm: bool = True,
        interpolate_mode: str = "bicubic",
        out_indices: Sequence[int] = (-1,),
        with_cls_token: bool = False,
        output_cls_token: bool = False,
        shift_pixel: int = 1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = depth
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.shift_pixel = shift_pixel

        # Patch embedding  Conv2d -> LayerNorm
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dims,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            ),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(embed_dims),
            Permute(0, 3, 1, 2),
        )

        h = w = img_size // patch_size
        self.patch_resolution = (h, w)
        num_patches = h * w

        # Position embedding
        self.num_extra_tokens = 1 if with_cls_token else 0
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))

        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        # Drop-path rates  (stochastic depth, linear schedule)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Vision_RWKV7_Block(
                    n_embd=embed_dims,
                    n_head=num_heads,
                    n_layer=depth,
                    layer_id=i,
                    drop_path=dpr[i],
                    init_values=init_values,
                    with_cls_token=with_cls_token,
                )
            )

        # Final norm
        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(embed_dims)

        # Resolve out_indices
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        out_indices = list(out_indices)
        for i, idx in enumerate(out_indices):
            if idx < 0:
                out_indices[i] = depth + idx
        self.out_indices = sorted(set(i for i in out_indices if 0 <= i < depth)) or [
            depth - 1
        ]

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.with_cls_token:
                self.cls_token.zero_()
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B = x.shape[0]

        # Patch embed:  [B, 3, H, W] -> [B, D, H', W']
        x = self.patch_embed(x)
        _, D, H, W = x.shape
        patch_resolution = (H, W)

        # Flatten: [B, D, H, W] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)

        # Position embedding (Feature 4: interpolatable for resolution change)
        pos_embed = resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens,
        )
        x = x + pos_embed.to(x.dtype)

        # CLS token  (post-position, matching VRWKV6)  [Feature 4 sub-feature]
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)

        # Blocks with multi-scale output
        outs = []
        vf_fwd, vf_bwd = None, None
        for i, block in enumerate(self.blocks):
            x, vff, vfb = block(x, patch_resolution, vf_fwd, vf_bwd)
            if i == 0:
                vf_fwd, vf_bwd = vff, vfb

            if i == len(self.blocks) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                if self.with_cls_token:
                    patch_tokens = x[:, :-1]
                    cls_out = x[:, -1]
                else:
                    patch_tokens = x
                    cls_out = None

                # [B, N, D] -> [B, D, H, W]
                feat = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)

                if self.output_cls_token and cls_out is not None:
                    outs.append((feat, cls_out))
                else:
                    outs.append(feat)

        return tuple(outs)
