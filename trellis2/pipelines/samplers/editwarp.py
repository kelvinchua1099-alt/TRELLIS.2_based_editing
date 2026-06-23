"""
EditWarp: multi-view velocity guidance for stable mask-free 3D editing.

Implements the multi-view velocity guidance of the EditWarp progress document
(Cai Yizhong, Sec. 1.5.2), built on top of the velocity-space VS3D editor
(:class:`FlowEditSampler`).

Motivation (EditWarp Sec. 1.2 / 1.4).  Plain FlowEdit defines an edit direction
from a *single* front-view velocity difference ``v_tgt - v_src``.  Under
occlusion the front view cannot see the edited region from every angle, so the
single-view edit signal is ambiguous and TRELLIS' own prior dominates -- the
edited geometry comes out fragmented (the "horn growing from a potted plant"
failure case).  Multi-view velocity guidance anchors the edit on the front view
and *refines* it with edit increments computed from additional rendered views,
fusing them in velocity space so the edit becomes multi-view consistent.

Because the TRELLIS 3D latent is view-independent (a single structured latent),
the only thing that changes per view is the *image condition*.  We therefore
share one running latent ``z_t`` across all views and, at every denoising step,
combine the per-view CFG velocities into a single fused edit velocity
(EditWarp Sec. 1.5.2, fusion rule):

    v = v_tgt^(1) + sum_{i>=2} w_i * ( v_tgt^(i) - v_src^(i) )

where view 1 is the anchor (front view) providing the base target velocity and
views ``i>=2`` contribute FlowEdit increments.  Setting the ``w_i`` to zero
recovers the single-view VS3D edit; a recurrence over views is supported by
feeding the fused velocity back as the anchor (``recurrent=True``).

The additional-view target images are supplied by a refine callback
(EditWarp Sec. 1.5.2, "diffusion/VLM refinement", Difix3D-style):

    refine_fn(rendered_view_i, first_edit_image, prompt) -> edited_view_i

with default ``prompt`` = "按照该图像,维持几何一致性地修复这张渲染图".
"""
from typing import *
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from .flow_edit import FlowEditSampler, VS3D_DEFAULTS


# Default refine prompt for additional views (EditWarp Sec. 1.5.2).
DEFAULT_REFINE_PROMPT = "按照该图像，维持几何一致性地修复这张渲染图"

# A refine callback maps (rendered_view, first_edit_image, prompt) -> edited view.
RefineFn = Callable[[Any, Any, str], Any]


class MultiViewFlowEditSampler(FlowEditSampler):
    """FlowEdit with multi-view velocity guidance (EditWarp Sec. 1.5.2).

    The conditions are passed as *lists*: ``cond_tgt_views[0]`` /
    ``cond_src_views[0]`` is the anchor (front) view, the rest are the refined
    additional views.  All other behaviour (RASI null-embedding, guidance
    interval, sigma_min coupling) is inherited from :class:`FlowEditSampler`.
    """

    # PLACEHOLDER_FUSE

    def multiview_velocity_delta(
        self,
        model,
        z_edit,
        x_src,
        t: float,
        cond_src_views: List[torch.Tensor],
        cond_tgt_views: List[torch.Tensor],
        phi,
        omega_src: float,
        omega_tgt: float,
        guidance_interval,
        num_noise: int,
        pmg_w: float,
        pmg_L: int,
        view_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        """Fused multi-view edit velocity (EditWarp Sec. 1.5.2 fusion rule).

        For each shared-noise coupling we compute, per view, the source and
        target CFG velocities on the *same* latent (only the image condition
        differs across views), then fuse:

            v = v_tgt^(1) + sum_{i>=2} w_i ( v_tgt^(i) - v_src^(i) )

        and average over ``num_noise`` couplings via the inherited PMG
        partial-mean amplification.  The anchor view (index 0) supplies the base
        target velocity; additional views contribute edit increments only.
        """
        n_views = len(cond_tgt_views)
        if view_weights is None:
            # default: anchor weight implicit (1.0 on base term), extra views = 1.0 each.
            view_weights = [1.0] * n_views
        assert len(cond_src_views) == n_views, "src/tgt view count mismatch"

        samples = []
        for _ in range(num_noise):
            eps = torch.randn_like(x_src)
            z_src = (1 - t) * x_src + (self.sigma_min + (1 - self.sigma_min) * t) * eps
            z_tgt = z_edit + (z_src - x_src)

            # anchor (view 0): base target velocity v_tgt^(1).
            v = self._cfg_velocity(
                model, z_tgt, t, cond_tgt_views[0], phi, omega_tgt,
                guidance_interval, **kwargs)
            # additional views: FlowEdit increments (v_tgt^(i) - v_src^(i)).
            for i in range(1, n_views):
                v_tgt_i = self._cfg_velocity(
                    model, z_tgt, t, cond_tgt_views[i], phi, omega_tgt,
                    guidance_interval, **kwargs)
                v_src_i = self._cfg_velocity(
                    model, z_src, t, cond_src_views[i], phi, omega_src,
                    guidance_interval, **kwargs)
                v = v + view_weights[i] * (v_tgt_i - v_src_i)
            samples.append(v)

        v_stack = torch.stack(samples, dim=0)
        mu_S = v_stack.mean(dim=0)
        L = max(1, min(pmg_L, num_noise - 1)) if num_noise > 1 else 1
        mu_L = v_stack[:L].mean(dim=0)
        return mu_S + pmg_w * (mu_S - mu_L)

    # PLACEHOLDER_EDIT

    @torch.no_grad()
    def edit_multiview(
        self,
        model,
        x_src,
        cond_src_views: List[torch.Tensor],
        cond_tgt_views: List[torch.Tensor],
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
        view_weights: Optional[List[float]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """Transport ``x_src`` to an edited latent under multi-view guidance.

        Mirrors :meth:`FlowEditSampler.edit` (RASI calibration -> PMG editing),
        but Phase 2 fuses per-view velocities.  RASI calibration uses the anchor
        view only (index 0) -- the source-reconstruction probe is view-agnostic
        once anchored, so a single view is sufficient and keeps it cheap.

        The recurrence over denoising steps is carried implicitly by the running
        latent ``z_edit`` integrated along the ODE (EditWarp's "实现递推").
        """
        anchor_src = cond_src_views[0]
        if neg_cond is None:
            neg_cond = torch.zeros_like(anchor_src)
        t_pairs = self._make_t_seq(steps, rescale_t)

        # ---- Phase 1: RASI calibration on the anchor view ----------------
        phi_cache: Dict[int, torch.Tensor] = {}
        z_edit = x_src.clone()
        for i, (t, t_prev) in enumerate(
            tqdm(t_pairs, desc="EditWarp RASI", disable=not verbose)
        ):
            if not (n_min <= i < n_max):
                continue
            eta_scale = 1.0 - i / max(1, len(t_pairs) - 1)
            phi = self.rasi_calibrate_step(
                model, z_edit, x_src, t, t_prev, anchor_src,
                phi_init=neg_cond, omega_tgt=omega_tgt,
                guidance_interval=guidance_interval,
                rasi_steps=rasi_steps, lr=rasi_lr,
                early_stop=rasi_early_stop, eta_scale=eta_scale, **kwargs,
            )
            phi_cache[i] = phi
            v_delta = self.pmg_velocity_delta(
                model, z_edit, x_src, t, anchor_src, anchor_src, phi,
                omega_src, omega_tgt, guidance_interval,
                num_noise=1, pmg_w=0.0, pmg_L=1, **kwargs,
            )
            z_edit = z_edit - (t - t_prev) * v_delta

        # ---- Phase 2: multi-view PMG editing -----------------------------
        z_edit = x_src.clone()
        for i, (t, t_prev) in enumerate(
            tqdm(t_pairs, desc="EditWarp multi-view", disable=not verbose)
        ):
            if not (n_min <= i < n_max):
                continue
            phi = phi_cache.get(i, neg_cond)
            v_delta = self.multiview_velocity_delta(
                model, z_edit, x_src, t,
                cond_src_views, cond_tgt_views, phi,
                omega_src, omega_tgt, guidance_interval,
                num_noise=num_noise, pmg_w=pmg_w, pmg_L=pmg_L,
                view_weights=view_weights, **kwargs,
            )
            z_edit = z_edit - (t - t_prev) * v_delta
        return edict({
            "samples": z_edit,
            "phi_cache": {str(k): v for k, v in phi_cache.items()},
        })
