"""
EditWarp multi-view velocity guidance example.

Two-pass mask-free 3D editing on a frozen TRELLIS 2.0 backbone:
  Pass 1: single-view VS3D anchor edit (RASI + PMG + TAR).
  Pass 2: render the anchor to several views, refine each extra view with a
          geometry-consistent 2D edit, then re-edit the occupancy with
          multi-view velocity guidance (EditWarp Sec. 1.5.2).

The refine callback is the integration point for your own 2D editor / VLM /
diffusion model.  Its contract is:

    refine_fn(rendered_view_i, first_edit_image, prompt) -> edited_view_i

where ``prompt`` defaults to
"按照该图像，维持几何一致性地修复这张渲染图".

Usage:
    python example_editwarp.py \
        --mesh assets/example_texturing/the_forgotten_knight.ply \
        --edited assets/example_image/T.png \
        --out edited_editwarp.glb
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
from trellis2.pipelines.edit_model_api import make_vlm_refine_fn, EditModelConfig
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel


def make_refine_fn(fallback_identity: bool = False):
    """构造 EditWarp 的 refine 回调,对接 VLM edit-model。

    API key 在 trellis2/pipelines/edit_model_api.py 的 EDIT_MODEL_API_KEY 占位处
    填写,或用环境变量 EDIT_MODEL_API_KEY / EDIT_MODEL_BASE_URL / EDIT_MODEL_NAME。
    请求体与返回解析的 TODO 也在该文件中,按你的 VLM provider 改两处即可。

    fallback_identity=True 时,key 缺失或调用失败会回退为返回原渲染图(便于在
    没配好 key 时先跑通流程调试)。
    """
    return make_vlm_refine_fn(EditModelConfig(), fallback_identity=fallback_identity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', required=True, help='Source 3D asset (.glb/.ply/.obj).')
    parser.add_argument('--edited', required=True, help='Front-view 2D-edited target image.')
    parser.add_argument('--source', default=None, help='Optional source-view image for c_src.')
    parser.add_argument('--out', default='edited_editwarp.glb')
    parser.add_argument('--pipeline_type', default='1024', choices=['512', '1024'])
    parser.add_argument('--nviews', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_refine', action='store_true',
                        help='跳过 VLM 细化(退化为单视图 VS3D)。')
    parser.add_argument('--refine_fallback', action='store_true',
                        help='VLM key 缺失或调用失败时回退为返回原渲染图(调试用)。')
    args = parser.parse_args()

    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))

    pipeline = Trellis2EditPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    mesh = trimesh.load(args.mesh)
    edited_image = Image.open(args.edited)
    source_image = Image.open(args.source) if args.source else None
    refine_fn = None if args.no_refine else make_refine_fn(fallback_identity=args.refine_fallback)

    edited_mesh = pipeline.run_editwarp(
        mesh=mesh,
        edited_image=edited_image,
        source_image=source_image,
        refine_fn=refine_fn,
        nviews=args.nviews,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
    )[0]
    edited_mesh.simplify(16777216)

    video = render_utils.make_pbr_vis_frames(render_utils.render_video(edited_mesh, envmap=envmap))
    imageio.mimsave("edited_editwarp_preview.mp4", video, fps=15)

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
        remesh=True, remesh_band=1, remesh_project=0, verbose=True,
    )
    glb.export(args.out, extension_webp=True)
    print(f"Saved EditWarp result to {args.out}")


if __name__ == '__main__':
    main()
