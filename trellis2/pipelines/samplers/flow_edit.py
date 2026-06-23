"""
Velocity-Space 3D Asset Editing (VS3D) sampler.

Implements the three modules of VS3D (Velocity-Space 3D Asset Editing) on top of
the TRELLIS 2.0 rectified-flow samplers:

  * RASI -- Reconstruction-Anchored Source Injection.  Per-timestep calibration
    of an unconditional (null) embedding ``phi_t`` so that the velocity
    difference ``v_delta`` collapses toward zero on non-edited regions, closing
    the two-channel identity leakage of plain FlowEdit (paper Sec. 3.2, Eq. 7).
  * PMG  -- Partial-Mean Guidance.  Amplifies the edit signal by extrapolating
    between a full-sample mean and a noisier partial-sample mean of ``v_delta``,
    automatically gated by the edit region itself (paper Sec. 3.3, Eq. 8-9).
  * TAR  -- Twin-Agreement Residual injection.  A condition-swapped twin forward
    on the sparse geometry / material stages decides, token by token, what to
    preserve via a per-token agreement map (paper Sec. 3.4, Eq. 10-11).  See
    ``twin_agreement_residual`` below; it is consumed by ``Trellis2EditPipeline``.

The sign conventions follow ``FlowEulerSampler``: timesteps run ``1 -> 0`` and a
single Euler step is ``x_{t_prev} = x_t - (t - t_prev) * v``.  All defaults are
the shared hyper-parameters reported in the paper's appendix (Sec. A.1).
"""
from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

from .flow_euler import FlowEulerSampler


# ---------------------------------------------------------------------------
# Default hyper-parameters (paper Sec. A.1, "Implementation details").
# ---------------------------------------------------------------------------
VS3D_DEFAULTS = edict(
    steps=25,                 # T : discretised timesteps
    n_max=12,                 # active window upper index (edit applied for step < n_max)
    n_min=0,                  # active window lower index (edit applied for step >= n_min)
    num_noise=5,              # S : Monte-Carlo noise samples per step for v_delta
    omega_src=1.5,            # CFG weight on the source branch
    omega_tgt=9.0,            # CFG weight on the target branch
    guidance_interval=(0.6, 1.0),   # CFG-active interval in t
    pmg_w=1.2,                # PMG extrapolation weight w (Eq. 8)
    pmg_L=2,                  # PMG partial-sample size L (1 <= L < S)
    rasi_steps=3,             # K : RASI inner optimisation steps
    rasi_lr=1e-5,             # inner-loop learning rate eta_0
    rasi_early_stop=1e-5,     # early-stop threshold tau_es on reconstruction loss
)


class FlowEditSampler(FlowEulerSampler):
    """FlowEdit coupling extended with RASI calibration and PMG amplification.

    This sampler operates on the Stage-1 *dense* sparse-structure latent, where
    the FlowEdit coupling is available (matched token support between source and
    target).  The sparse Stage-2 / Stage-3 SLATs instead use
    :meth:`twin_agreement_residual` (TAR), since their active coordinates differ
    between source and target.
    """

    # -- helpers -----------------------------------------------------------
    def _make_t_seq(self, steps: int, rescale_t: float) -> List[Tuple[float, float]]:
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        return [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

    def _cfg_velocity(self, model, x_t, t, cond, neg_cond, omega, guidance_interval, **kwargs):
        """Classifier-free-guided velocity, gated by the guidance interval.

        Outside the interval the unconditional branch is dropped (omega = 1),
        matching ``GuidanceIntervalSamplerMixin``.
        """
        if not (guidance_interval[0] <= t <= guidance_interval[1]):
            return self._inference_model(model, x_t, t, cond, **kwargs)
        pred_pos = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_neg = self._inference_model(model, x_t, t, neg_cond, **kwargs)
        return omega * pred_pos + (1 - omega) * pred_neg

    # -- RASI : Reconstruction-Anchored Source Injection -------------------
    def rasi_calibrate_step(
        self,
        model,
        z_edit,
        x_src,
        t: float,
        t_prev: float,
        cond_src,
        phi_init,
        omega_tgt: float,
        guidance_interval,
        rasi_steps: int,
        lr: float,
        early_stop: float,
        eta_scale: float = 1.0,
        **kwargs,
    ):
        """Calibrate the null embedding ``phi_t`` at one timestep (Eq. 7).

        Both branches of the FlowEdit coupling are conditioned on the *source*
        condition ``c_src`` and keep their real editing-time guidance weight, so
        a single Euler step of ``v_delta`` is asked to reconstruct ``x_src``.
        The optimised ``phi`` absorbs both leakage channels into one per-step,
        asset-specific correction.

        Returns the calibrated ``phi`` (detached) for caching.
        """
        dt = t - t_prev
        # shared-noise coupling (Eq. 3) on the interpolation of x_src.
        eps = torch.randn_like(x_src)
        z_src = (1 - t) * x_src + (self.sigma_min + (1 - self.sigma_min) * t) * eps
        # z_tgt tracks the running edit offset: z_tgt = z_t^edit + (z_src - x_src).
        z_tgt = z_edit + (z_src - x_src)

        phi = phi_init.detach().clone().requires_grad_(True)
        optim = torch.optim.Adam([phi], lr=lr * eta_scale)

        with torch.enable_grad():
            for _ in range(rasi_steps):
                optim.zero_grad()
                # target branch: c_src condition, phi as null, omega_tgt weight.
                v_tgt = self._cfg_velocity(
                    model, z_tgt, t, cond_src, phi, omega_tgt, guidance_interval, **kwargs)
                # source branch: c_src condition, phi as null, omega_tgt weight.
                v_src = self._cfg_velocity(
                    model, z_src, t, cond_src, phi, omega_tgt, guidance_interval, **kwargs)
                v_delta = v_tgt - v_src
                # one Euler step on the probe ODE should bring z_tgt back to x_src.
                z_rec = z_tgt - dt * v_delta
                loss = torch.nn.functional.mse_loss(z_rec, x_src)
                loss.backward()
                optim.step()
                if loss.item() < early_stop:
                    break

        return phi.detach()

    # -- PMG : Partial-Mean Guidance ---------------------------------------
    def pmg_velocity_delta(
        self,
        model,
        z_edit,
        x_src,
        t: float,
        cond_src,
        cond_tgt,
        phi,
        omega_src: float,
        omega_tgt: float,
        guidance_interval,
        num_noise: int,
        pmg_w: float,
        pmg_L: int,
        **kwargs,
    ):
        """Edit velocity ``v_delta`` amplified by partial-mean guidance (Eq. 8).

        ``v_delta`` is estimated as a Monte-Carlo average over ``S`` shared-noise
        couplings.  We extrapolate away from a noisier ``L``-sample mean toward
        the full ``S``-sample mean: ``mu_S + w (mu_S - mu_L)``.  The cached RASI
        ``phi`` is substituted for the network null embedding in every CFG call,
        so identity preservation on non-edited voxels carries through.
        """
        samples = []
        for _ in range(num_noise):
            eps = torch.randn_like(x_src)
            z_src = (1 - t) * x_src + (self.sigma_min + (1 - self.sigma_min) * t) * eps
            z_tgt = z_edit + (z_src - x_src)
            # target branch uses c_tgt; source branch uses c_src; both use phi as null.
            v_tgt = self._cfg_velocity(
                model, z_tgt, t, cond_tgt, phi, omega_tgt, guidance_interval, **kwargs)
            v_src = self._cfg_velocity(
                model, z_src, t, cond_src, phi, omega_src, guidance_interval, **kwargs)
            samples.append(v_tgt - v_src)

        v_stack = torch.stack(samples, dim=0)          # [S, ...]
        mu_S = v_stack.mean(dim=0)                       # full-sample mean
        L = max(1, min(pmg_L, num_noise - 1))
        mu_L = v_stack[:L].mean(dim=0)                   # partial-sample mean
        return mu_S + pmg_w * (mu_S - mu_L)              # Eq. 8

    # -- main editing loop (RASI calibration -> PMG editing) ---------------
    @torch.no_grad()
    def edit(
        self,
        model,
        x_src,
        cond_src,
        cond_tgt,
        neg_cond=None,
        steps: int = VS3D_DEFAULTS.steps,
        rescale_t: float = 1.0,
        n_max: int = VS3D_DEFAULTS.n_max,
        n_min: int = VS3D_DEFAULTS.n_min,
        num_noise: int = VS3D_DEFAULTS.num_noise,
        omega_src: float = VS3D_DEFAULTS.omega_src,
        omega_tgt: float = VS3D_DEFAULTS.omega_tgt,
        guidance_interval: Tuple[float, float] = VS3D_DEFAULTS.guidance_interval,
        pmg_w: float = VS3D_DEFAULTS.pmg_w,
        pmg_L: int = VS3D_DEFAULTS.pmg_L,
        rasi_steps: int = VS3D_DEFAULTS.rasi_steps,
        rasi_lr: float = VS3D_DEFAULTS.rasi_lr,
        rasi_early_stop: float = VS3D_DEFAULTS.rasi_early_stop,
        verbose: bool = True,
        **kwargs,
    ):
        """Transport ``x_src`` to an edited Stage-1 latent in velocity space.

        Runs Algorithm 1, Stage 1: a RASI calibration pass that caches a
        per-timestep ``phi_t``, followed by a PMG editing pass that integrates
        the amplified edit offset.  The edit is only applied inside the active
        window ``n_min <= step < n_max``; outside it the latent is left untouched.
        """
        if neg_cond is None:
            neg_cond = torch.zeros_like(cond_src)
        t_pairs = self._make_t_seq(steps, rescale_t)

        # ---- Phase 1: RASI calibration, cache phi_t ----------------------
        phi_cache: Dict[int, torch.Tensor] = {}
        z_edit = x_src.clone()
        for i, (t, t_prev) in enumerate(
            tqdm(t_pairs, desc="VS3D RASI", disable=not verbose)
        ):
            if not (n_min <= i < n_max):
                continue
            eta_scale = 1.0 - i / max(1, len(t_pairs) - 1)   # linear anneal of eta_t
            phi = self.rasi_calibrate_step(
                model, z_edit, x_src, t, t_prev, cond_src,
                phi_init=neg_cond, omega_tgt=omega_tgt,
                guidance_interval=guidance_interval,
                rasi_steps=rasi_steps, lr=rasi_lr,
                early_stop=rasi_early_stop, eta_scale=eta_scale, **kwargs,
            )
            phi_cache[i] = phi
            # advance z_edit by one Euler step on the source-reconstruction ODE.
            v_delta = self.pmg_velocity_delta(
                model, z_edit, x_src, t, cond_src, cond_src, phi,
                omega_src, omega_tgt, guidance_interval,
                num_noise=1, pmg_w=0.0, pmg_L=1, **kwargs,
            )
            z_edit = z_edit - (t - t_prev) * v_delta

        # ---- Phase 2: PMG editing ----------------------------------------
        z_edit = x_src.clone()
        for i, (t, t_prev) in enumerate(
            tqdm(t_pairs, desc="VS3D PMG", disable=not verbose)
        ):
            if not (n_min <= i < n_max):
                continue
            phi = phi_cache.get(i, neg_cond)
            v_delta = self.pmg_velocity_delta(
                model, z_edit, x_src, t, cond_src, cond_tgt, phi,
                omega_src, omega_tgt, guidance_interval,
                num_noise=num_noise, pmg_w=pmg_w, pmg_L=pmg_L, **kwargs,
            )
            z_edit = z_edit - (t - t_prev) * v_delta

        # phi_cache is kept with string keys so it survives edict construction.
        return edict({"samples": z_edit, "phi_cache": {str(k): v for k, v in phi_cache.items()}})

    # PLACEHOLDER_TAR


# ---------------------------------------------------------------------------
# TAR : Twin-Agreement Residual injection (paper Sec. 3.4, Eq. 10-11).
# ---------------------------------------------------------------------------
def twin_agreement_pkeep(
    z_tgt: torch.Tensor,
    z_src_twin: torch.Tensor,
    alpha: float = 0.05,
    beta: float = 0.95,
) -> torch.Tensor:
    """Per-token preserve-confidence ``p_keep`` from a condition-swapped twin.

    Given the target-conditioned forward ``z_tgt`` and the source-conditioned
    twin ``z_src_twin`` (same scaffold + seeded noise, only the image embedding
    differs), the per-token disagreement ``d_i = ||z_tgt[i] - z_src_twin[i]||_2``
    is mapped to a preserve-confidence via robust quantile clipping (Eq. 10):

        p_keep[i] = 1 - clip((d_i - q_alpha(d)) / (q_beta(d) - q_alpha(d)), 0, 1)

    Tokens that look the same under both conditions get ``p_keep ~ 1`` and are
    safe to preserve; edit-sensitive tokens diverge and get ``p_keep ~ 0``.

    Args:
        z_tgt: [N, C] target-conditioned token features.
        z_src_twin: [N, C] source-conditioned twin token features.
    Returns:
        [N] preserve-confidence in [0, 1].
    """
    d = torch.linalg.vector_norm(z_tgt - z_src_twin, dim=-1)   # [N]
    q_a = torch.quantile(d, alpha)
    q_b = torch.quantile(d, beta)
    denom = torch.clamp(q_b - q_a, min=1e-8)
    p_keep = 1.0 - torch.clamp((d - q_a) / denom, 0.0, 1.0)
    return p_keep


def twin_agreement_residual(
    z_tgt: torch.Tensor,
    z_src_enc: torch.Tensor,
    p_keep: torch.Tensor,
    intersection_mask: torch.Tensor,
    lam: float = 0.5,
    tau: float = 10.0,
    theta: float = 0.7,
) -> torch.Tensor:
    """Blend a norm-clipped source residual into the target latent (Eq. 11).

    On the integer-voxel intersection ``I = C_tgt ∩ C_src`` we form the residual
    ``r_i = clip_tau(z_src_enc[i] - z_tgt[i])`` and blend:

        z[i] = z_tgt[i] + lam * p_keep[i] * 1[p_keep[i] >= theta] * r_i   (i in I)
        z[i] = z_tgt[i]                                                   (otherwise)

    Edit-only tokens (outside the intersection) stay on the target branch,
    protecting freshly generated geometry / material; agreeing tokens are softly
    retracted toward the source encoding with strength governed by ``p_keep``.

    Args:
        z_tgt: [N, C] target-branch latent (modified in a copy).
        z_src_enc: [N, C] SC-VAE source encoding, aligned to ``z_tgt`` tokens;
            rows outside the intersection are ignored.
        p_keep: [N] preserve-confidence from :func:`twin_agreement_pkeep`.
        intersection_mask: [N] bool, True where the token's voxel also exists in
            the source asset (i.e. lies in ``I``).
    Returns:
        [N, C] blended latent.
    """
    z = z_tgt.clone()
    if intersection_mask.sum() == 0:
        return z
    r = z_src_enc - z_tgt
    # norm-clip the residual to radius tau (per-token).
    r_norm = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
    scale = torch.clamp(tau / torch.clamp(r_norm, min=1e-8), max=1.0)
    r = r * scale
    gate = (p_keep >= theta).float() * p_keep            # zeroed where twin disagrees
    weight = (lam * gate * intersection_mask.float()).unsqueeze(-1)
    z = z + weight * r
    return z
