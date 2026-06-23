"""
VS3D local 3D editing example.

Edits an existing 3D asset toward a single 2D-edited target image using the
velocity-space VS3D pipeline (RASI + PMG + TAR) on a frozen TRELLIS 2.0 backbone.

Usage:
    python example_editing.py \
        --mesh assets/example_texturing/the_forgotten_knight.ply \
        --edited assets/example_image/T.png \
        --out edited.glb
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import cv2
import imageio
import trimesh
from PIL import Image
import torch

from trellis2.pipelines import Trellis2EditPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', required=True, help='Source 3D asset (.glb/.ply/.obj).')
    parser.add_argument('--edited', required=True, help='2D-edited target image.')
    parser.add_argument('--source', default=None, help='Optional source-view image for c_src.')
    parser.add_argument('--out', default='edited.glb')
    parser.add_argument('--pipeline_type', default='1024', choices=['512', '1024'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 1. Environment map for PBR visualisation.
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))

    # 2. Load the editing pipeline.
    pipeline = Trellis2EditPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    # 3. Load source asset + edited image, then run VS3D editing.
    mesh = trimesh.load(args.mesh)
    edited_image = Image.open(args.edited)
    source_image = Image.open(args.source) if args.source else None

    edited_mesh = pipeline.run(
        mesh=mesh,
        edited_image=edited_image,
        source_image=source_image,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
    )[0]
    edited_mesh.simplify(16777216)  # nvdiffrast limit

    # 4. Render a preview video.
    video = render_utils.make_pbr_vis_frames(render_utils.render_video(edited_mesh, envmap=envmap))
    imageio.mimsave("edited_preview.mp4", video, fps=15)

    # 5. Export to GLB.
    glb = o_voxel.postprocess.to_glb(
        vertices=edited_mesh.vertices,
        faces=edited_mesh.faces,
        attr_volume=edited_mesh.attrs,
        coords=edited_mesh.coords,
        attr_layout=edited_mesh.layout,
        voxel_size=edited_mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(args.out, extension_webp=True)
    print(f"Saved edited asset to {args.out}")


if __name__ == '__main__':
    main()
