"""
VS3D editing pipeline for TRELLIS 2.0.

``Trellis2EditPipeline`` performs training-free, mask-free local editing of an
existing 3D asset by intervening inside the rectified-flow ODE of a frozen
TRELLIS 2.0 generator, following the VS3D paper (Velocity-Space 3D Asset
Editing).  Given an original 3D asset and a single 2D-edited target image it:

  Stage 1 (dense occupancy latent): edits with RASI + PMG via
    :class:`FlowEditSampler`, transporting the source latent ``x_src`` to an
    edited occupancy in velocity space.
  Stage 2 / 3 (sparse geometry / material SLATs): regenerates under the target
    condition, then applies TAR (twin-agreement residual injection) to retract
    non-edited tokens toward the SC-VAE source encoding.

The source condition ``c_src`` is obtained by rendering the input asset to a
canonical view; the edited image provides the target condition ``c_tgt``.
"""
from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import trimesh

from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from .samplers.flow_edit import (
    FlowEditSampler, VS3D_DEFAULTS,
    twin_agreement_pkeep, twin_agreement_residual,
)
from ..modules.sparse import SparseTensor
import o_voxel


class Trellis2EditPipeline(Trellis2ImageTo3DPipeline):
    """Image-conditioned local 3D editing on a frozen TRELLIS 2.0 backbone."""

    # Reuse the generative DiTs + decoders, and add the encoders needed to map
    # the source asset into latent space (x_src and SC-VAE source encodings).
    model_names_to_load = Trellis2ImageTo3DPipeline.model_names_to_load + [
        'sparse_structure_encoder',
        'shape_slat_encoder',
        'tex_slat_encoder',
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A dedicated FlowEdit sampler reused across the Stage-1 edit.
        self.edit_sampler = FlowEditSampler(
            sigma_min=getattr(self.sparse_structure_sampler, 'sigma_min', 1e-5)
        )

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2EditPipeline":
        pipeline = super().from_pretrained(path, config_file)
        pipeline.edit_sampler = FlowEditSampler(
            sigma_min=getattr(pipeline.sparse_structure_sampler, 'sigma_min', 1e-5)
        )
        return pipeline

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def render_source_condition(self, mesh) -> Image.Image:
        """Render the source asset to a canonical front view for ``c_src``.

        The render is background-removed and framed exactly like the user image
        path so that ``c_src`` and ``c_tgt`` live in the same conditioning space.
        """
        from ..utils import render_utils
        frames = render_utils.render_snapshot([mesh], resolution=512, nviews=1)
        img = frames['color'][0] if isinstance(frames, dict) else frames[0]
        if isinstance(img, torch.Tensor):
            img = (img.clamp(0, 1) * 255).byte().cpu().numpy()
        return Image.fromarray(img)

    # ------------------------------------------------------------------
    # Source encoding
    # ------------------------------------------------------------------
    def encode_source_occupancy(self, mesh: trimesh.Trimesh, ss_res: int = 64) -> torch.Tensor:
        """Encode the source mesh into the Stage-1 dense latent ``x_src``.

        The mesh is voxelised to a binary occupancy grid at the sparse-structure
        decoder resolution, then passed through the frozen ``SparseStructureEncoder``
        to obtain the dense latent the Stage-1 DiT operates on.
        """
        # Voxelise the surface to an occupancy grid (dense [1, 1, R, R, R]).
        verts = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        occ_coords = o_voxel.convert.mesh_to_flexible_dual_grid(
            verts.cpu(), faces.cpu(),
            grid_size=ss_res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0, boundary_weight=0.2,
            regularization_weight=1e-2, timing=False,
        )[0]  # voxel_indices [M, 3]
        occ = torch.zeros(1, 1, ss_res, ss_res, ss_res)
        idx = occ_coords.long()
        occ[0, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        occ = occ.to(self.device)

        encoder = self.models['sparse_structure_encoder']
        if self.low_vram:
            encoder.to(self.device)
        x_src = encoder(occ, sample_posterior=False)
        if self.low_vram:
            encoder.cpu()
        return x_src

    def encode_source_shape_slat(self, mesh: trimesh.Trimesh, resolution: int) -> SparseTensor:
        """SC-VAE geometry encoding of the source asset (``z_src^enc`` for TAR)."""
        verts = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            verts.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0, boundary_weight=0.2,
            regularization_weight=1e-2, timing=False,
        )
        vertices = SparseTensor(
            feats=dual_vertices * resolution - voxel_indices,
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1),
        ).to(self.device)
        intersected = vertices.replace(intersected).to(self.device)
        enc = self.models['shape_slat_encoder']
        if self.low_vram:
            enc.to(self.device)
        z = enc(vertices, intersected)
        if self.low_vram:
            enc.cpu()
        return z

    def encode_source_tex_slat(self, mesh: trimesh.Trimesh, resolution: int) -> Optional[SparseTensor]:
        """SC-VAE material encoding of the source asset (``z_src^enc`` for TAR).

        Returns ``None`` when no material SC-VAE encoder is available in the
        loaded model set (some released checkpoints ship decoder-only material
        VAEs); the caller then skips material-stage residual injection.
        """
        if 'tex_slat_encoder' not in self.models:
            return None
        coord, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
            mesh, grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        )
        # Assemble the 6-channel PBR feature in the pipeline's attr layout and
        # normalise byte attributes (0..255) to [0, 1].
        feats = torch.cat([
            attr['base_color'].float() / 255.0,   # 3
            attr['metallic'].float() / 255.0,      # 1
            attr['roughness'].float() / 255.0,     # 1
            attr['alpha'].float() / 255.0,         # 1
        ], dim=-1)
        src = SparseTensor(
            feats=feats,
            coords=torch.cat([torch.zeros_like(coord[:, 0:1]), coord], dim=-1),
        ).to(self.device)
        enc = self.models['tex_slat_encoder']
        if self.low_vram:
            enc.to(self.device)
        z = enc(src)
        if self.low_vram:
            enc.cpu()
        return z

    # ------------------------------------------------------------------
    # TAR application on a SLAT stage
    # ------------------------------------------------------------------
    def apply_tar(
        self,
        z_tgt: SparseTensor,
        z_src_twin: SparseTensor,
        z_src_enc: Optional[SparseTensor],
        tar_params: dict,
    ) -> SparseTensor:
        """Apply Twin-Agreement Residual injection to one SLAT (Eq. 10-11).

        ``z_tgt`` and ``z_src_twin`` share the same coordinates (the target
        scaffold C_tgt) since the twin only swaps the image condition.  The
        source encoding ``z_src_enc`` lives on its own coordinates C_src; we
        align it to C_tgt by integer-voxel intersection and inject the residual
        only on agreeing tokens of that intersection.
        """
        p_keep = twin_agreement_pkeep(
            z_tgt.feats, z_src_twin.feats,
            alpha=tar_params['alpha'], beta=tar_params['beta'],
        )
        N, C = z_tgt.feats.shape
        if z_src_enc is None:
            return z_tgt
        # Build C_tgt <- C_src coordinate alignment (integer voxels).
        tgt_coords = z_tgt.coords[:, 1:].long()
        src_coords = z_src_enc.coords[:, 1:].long()
        # hash 3D integer coords to a 1D key for matching.
        res = int(tgt_coords.max().item()) + 2 if tgt_coords.numel() else 1
        def _key(c):
            return (c[:, 0] * res + c[:, 1]) * res + c[:, 2]
        tgt_key = _key(tgt_coords)
        src_key = _key(src_coords)
        src_lookup = {int(k): i for i, k in enumerate(src_key.tolist())}
        aligned = torch.zeros(N, C, device=z_tgt.feats.device, dtype=z_tgt.feats.dtype)
        mask = torch.zeros(N, dtype=torch.bool, device=z_tgt.feats.device)
        for i, k in enumerate(tgt_key.tolist()):
            j = src_lookup.get(int(k))
            if j is not None:
                aligned[i] = z_src_enc.feats[j]
                mask[i] = True
        blended = twin_agreement_residual(
            z_tgt.feats, aligned, p_keep, mask,
            lam=tar_params['lam'], tau=tar_params['tau'], theta=tar_params['theta'],
        )
        return z_tgt.replace(blended)

    # ------------------------------------------------------------------
    # Stage-1 occupancy editing + decode to coords
    # ------------------------------------------------------------------
    @torch.no_grad()
    def edit_sparse_structure(
        self,
        x_src: torch.Tensor,
        cond_src: dict,
        cond_tgt: dict,
        resolution: int,
        edit_params: dict,
    ) -> torch.Tensor:
        """Edit the dense occupancy latent (RASI + PMG) and decode to coords."""
        flow_model = self.models['sparse_structure_flow_model']
        if self.low_vram:
            flow_model.to(self.device)
        out = self.edit_sampler.edit(
            flow_model, x_src,
            cond_src=cond_src['cond'], cond_tgt=cond_tgt['cond'],
            neg_cond=cond_tgt.get('neg_cond'),
            **edit_params,
        )
        z_edit = out.samples
        if self.low_vram:
            flow_model.cpu()

        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_edit) > 0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        return coords

    # ------------------------------------------------------------------
    # SLAT sampling with a condition-swapped twin (TAR Stage 2/3)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_shape_slat_twin(self, cond_tgt, cond_src, flow_model, coords, sampler_params, seed):
        """Sample geometry SLAT under c_tgt and a c_src twin on the same seed."""
        in_ch = flow_model.in_channels
        torch.manual_seed(seed)
        noise_feats = torch.randn(coords.shape[0], in_ch).to(self.device)
        noise = SparseTensor(feats=noise_feats, coords=coords)
        sp_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_tgt = self.shape_slat_sampler.sample(
            flow_model, noise, **cond_tgt, **sp_params, verbose=True,
            tqdm_desc="Sampling geometry SLat (tgt)").samples
        # twin: identical seeded noise + scaffold, source condition.
        noise_twin = SparseTensor(feats=noise_feats.clone(), coords=coords)
        z_src_twin = self.shape_slat_sampler.sample(
            flow_model, noise_twin, **cond_src, **sp_params, verbose=True,
            tqdm_desc="Sampling geometry SLat (twin)").samples
        if self.low_vram:
            flow_model.cpu()
        return z_tgt, z_src_twin

    @torch.no_grad()
    def run(
        self,
        mesh: trimesh.Trimesh,
        edited_image: Image.Image,
        source_image: Optional[Image.Image] = None,
        seed: int = 42,
        pipeline_type: str = '1024',
        preprocess_image: bool = True,
        edit_params: Optional[dict] = None,
        tar_params: Optional[dict] = None,
    ):
        """Run the full VS3D three-stage editing pipeline (Algorithm 1).

        Args:
            mesh: the original 3D asset to edit (geometry, optionally textured).
            edited_image: the 2D-edited target image -> target condition c_tgt.
            source_image: optional pre-rendered source view -> c_src.  If omitted
                the source asset is rendered to a canonical view automatically.
            pipeline_type: SLAT resolution path, '512' or '1024'.
            edit_params: overrides for the Stage-1 RASI+PMG sampler (see
                ``VS3D_DEFAULTS``).
            tar_params: overrides for Stage-2/3 TAR
                (lam, tau, theta, alpha, beta).
        """
        edit_params = {**VS3D_DEFAULTS, **(edit_params or {})}
        # drop occupancy-only fields not accepted by the sampler? all accepted.
        tar_params = {
            'lam': 0.5, 'tau': 10.0, 'theta': 0.7, 'alpha': 0.05, 'beta': 0.95,
            **(tar_params or {}),
        }
        resolution = 512 if pipeline_type == '512' else 1024
        cond_res = 512 if pipeline_type == '512' else 1024

        # --- conditioning ------------------------------------------------
        if source_image is None:
            source_image = self.render_source_condition(mesh)
        if preprocess_image:
            edited_image = self.preprocess_image(edited_image)
            source_image = self.preprocess_image(source_image)
        cond_src = self.get_cond([source_image], cond_res)
        cond_tgt = self.get_cond([edited_image], cond_res)

        # --- Stage 1: occupancy editing (RASI + PMG) ---------------------
        x_src = self.encode_source_occupancy(mesh, ss_res=64)
        ss_res = {'512': 32, '1024': 64}[pipeline_type]
        coords = self.edit_sparse_structure(
            x_src, cond_src, cond_tgt, ss_res, edit_params)

        # --- Stage 2: geometry SLAT (TAR) --------------------------------
        shape_model = self.models[f'shape_slat_flow_model_{resolution}']
        z_geo_tgt, z_geo_twin = self.sample_shape_slat_twin(
            cond_tgt, cond_src, shape_model, coords, {}, seed)
        z_geo_src = self.encode_source_shape_slat(mesh, resolution)
        z_geo = self.apply_tar(z_geo_tgt, z_geo_twin, z_geo_src, tar_params)
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(z_geo.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(z_geo.device)
        shape_slat = z_geo * std + mean

        # --- Stage 3: material SLAT (TAR), geometry-conditioned ----------
        tex_model = self.models[f'tex_slat_flow_model_{resolution}']
        z_tex_tgt, z_tex_twin = self.sample_tex_slat_twin(
            cond_tgt, cond_src, tex_model, shape_slat, seed)
        z_tex_src = self.encode_source_tex_slat(mesh, resolution)
        z_tex = self.apply_tar(z_tex_tgt, z_tex_twin, z_tex_src, tar_params)
        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(z_tex.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(z_tex.device)
        tex_slat = z_tex * std + mean

        torch.cuda.empty_cache()
        return self.decode_latent(shape_slat, tex_slat, resolution)

    # PLACEHOLDER_TEXTWIN

    @torch.no_grad()
    def sample_tex_slat_twin(self, cond_tgt, cond_src, flow_model, shape_slat, seed):
        """Sample material SLAT under c_tgt and a c_src twin, geometry-conditioned.

        The (un-normalised) ``shape_slat`` is re-normalised and concatenated as
        the geometry condition, exactly as in the base texturing path; the twin
        reuses the same seeded noise and geometry, swapping only the image cond.
        """
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        concat_cond = (shape_slat - mean) / std

        in_ch = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        torch.manual_seed(seed)
        noise_feats = torch.randn(concat_cond.coords.shape[0], in_ch - concat_cond.feats.shape[1]).to(self.device)
        sp_params = {**self.tex_slat_sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_tgt = self.tex_slat_sampler.sample(
            flow_model, concat_cond.replace(noise_feats), concat_cond=concat_cond,
            **cond_tgt, **sp_params, verbose=True, tqdm_desc="Sampling material SLat (tgt)").samples
        z_src_twin = self.tex_slat_sampler.sample(
            flow_model, concat_cond.replace(noise_feats.clone()), concat_cond=concat_cond,
            **cond_src, **sp_params, verbose=True, tqdm_desc="Sampling material SLat (twin)").samples
        if self.low_vram:
            flow_model.cpu()
        return z_tgt, z_src_twin
