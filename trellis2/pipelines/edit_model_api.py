"""
VLM 图像编辑后端 —— 为 EditWarp 多视图细化提供 refine 回调。

EditWarp 的多视图速度引导(editwarp.py, Sec. 1.5.2)需要一个 refine 回调:
对每个额外视角,根据"该视角的渲染图 + 第一次编辑图 + prompt"生成一张
几何一致的修改图。本模块把这个回调对接到外部 VLM / 图像编辑模型,并把
API key 集中放在一个显眼的占位处。

用法:
    from trellis2.pipelines.edit_model_api import make_vlm_refine_fn, EditModelConfig
    refine_fn = make_vlm_refine_fn(EditModelConfig(api_key="sk-..."))
    pipeline.run_editwarp(..., refine_fn=refine_fn)

回调契约(与 editwarp.RefineFn 一致):
    refine_fn(rendered_view, first_edit_image, prompt) -> edited_view(PIL.Image)
"""
from typing import *
import os
import io
import base64
from dataclasses import dataclass, field
from PIL import Image


# ===========================================================================
#  API KEY 占位区
#  —— 把你的 edit-model API key 填到下面的占位,或设置环境变量 EDIT_MODEL_API_KEY。
#     环境变量优先级高于这里的占位常量。
# ===========================================================================
EDIT_MODEL_API_KEY = "<YOUR_EDIT_MODEL_API_KEY_HERE>"     # <-- 在此填入真实 API key

# 默认 endpoint 与模型名(按你的 VLM provider 修改,或用环境变量覆盖)
DEFAULT_BASE_URL = "<YOUR_EDIT_MODEL_BASE_URL_HERE>"      # 例:https://api.openai.com/v1
DEFAULT_MODEL_NAME = "<YOUR_EDIT_MODEL_NAME_HERE>"        # 例:gpt-image-1 / gemini-2.x / 自建模型名


@dataclass
class EditModelConfig:
    """edit-model(VLM 图像编辑)接口配置。

    api_key / base_url / model 都优先读环境变量,读不到再回退到上面的占位常量,
    方便你在不改代码的情况下用环境变量注入密钥。
    """
    api_key: str = field(default_factory=lambda: os.environ.get("EDIT_MODEL_API_KEY", EDIT_MODEL_API_KEY))
    base_url: str = field(default_factory=lambda: os.environ.get("EDIT_MODEL_BASE_URL", DEFAULT_BASE_URL))
    model: str = field(default_factory=lambda: os.environ.get("EDIT_MODEL_NAME", DEFAULT_MODEL_NAME))
    timeout: float = 120.0
    max_retries: int = 2

    def validate(self) -> None:
        if not self.api_key or self.api_key.startswith("<"):
            raise ValueError(
                "edit-model API key 未设置:请在 edit_model_api.py 的 EDIT_MODEL_API_KEY "
                "占位处填入真实 key,或设置环境变量 EDIT_MODEL_API_KEY。"
            )
        if not self.base_url or self.base_url.startswith("<"):
            raise ValueError("edit-model base_url 未设置(EDIT_MODEL_BASE_URL 或占位常量)。")


# ---------------------------------------------------------------------------
#  图像 <-> base64 编解码工具
# ---------------------------------------------------------------------------
def _image_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ---------------------------------------------------------------------------
#  实际 VLM 调用(占位实现 —— 按你的 provider 填充请求/解析逻辑)
# ---------------------------------------------------------------------------
def _call_edit_model(
    cfg: EditModelConfig,
    rendered_view: Image.Image,
    first_edit_image: Image.Image,
    prompt: str,
) -> Image.Image:
    """调用外部 edit-model 生成一张修改图。

    这里给出一个与 OpenAI 风格 chat/completions(多模态)兼容的占位实现:把
    两张图以 base64 data-url 传入,prompt 指示模型"按参考图、保持几何一致地修复
    渲染图"。不同 provider 的请求体/返回字段不一样,**请按你的 VLM 接口改 payload
    和返回解析这两处**(已用 TODO 标出)。
    """
    cfg.validate()
    try:
        import requests
    except ImportError as e:
        raise ImportError("需要 requests:pip install requests") from e

    rendered_b64 = _image_to_b64(rendered_view)
    ref_b64 = _image_to_b64(first_edit_image)

    # ---- TODO(1): 按你的 VLM provider 构造请求体 -------------------------
    payload = {
        "model": cfg.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rendered_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    url = cfg.base_url.rstrip("/") + "/chat/completions"

    last_err = None
    for _ in range(max(1, cfg.max_retries)):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout)
            resp.raise_for_status()
            data = resp.json()
            # ---- TODO(2): 按你的 VLM 返回结构解析出图像 base64 ----------
            # 占位:假设返回里带一个 base64 图像字段。请改成你接口真实的路径。
            img_b64 = data["choices"][0]["message"].get("image_b64")
            if not img_b64:
                raise ValueError("返回中未找到图像字段 image_b64,请按你的 provider 修改解析逻辑。")
            return _b64_to_image(img_b64)
        except Exception as e:  # noqa: BLE001 —— 网络/解析错误重试
            last_err = e
    raise RuntimeError(f"edit-model 调用失败(已重试 {cfg.max_retries} 次):{last_err}")


# ---------------------------------------------------------------------------
#  工厂:生成 EditWarp 用的 refine 回调
# ---------------------------------------------------------------------------
def make_vlm_refine_fn(
    config: Optional[EditModelConfig] = None,
    *,
    fallback_identity: bool = False,
) -> Callable[[Image.Image, Image.Image, str], Image.Image]:
    """构造 EditWarp 的 refine 回调,内部对接 VLM edit-model。

    Args:
        config: 接口配置;默认从环境变量 / 占位常量读取 api_key 等。
        fallback_identity: True 时,若 API key 缺失或调用失败,则回退为直接返回
            原渲染图(便于在没配好 key 时跑通流程做调试);False 时直接抛错。

    Returns:
        refine_fn(rendered_view, first_edit_image, prompt) -> edited_view
    """
    cfg = config or EditModelConfig()

    def refine_fn(rendered_view: Image.Image, first_edit_image: Image.Image, prompt: str) -> Image.Image:
        try:
            return _call_edit_model(cfg, rendered_view, first_edit_image, prompt)
        except Exception:  # noqa: BLE001
            if fallback_identity:
                return rendered_view
            raise

    return refine_fn
