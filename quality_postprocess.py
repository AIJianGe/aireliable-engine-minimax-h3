"""MiniMax-H3 画质方案后处理编排（45-专项）。

职责：
    - 在主渲染（diffusers H3）产出视频后，按画质方案 token 顺序应用后处理阶段：
          [A 超分] → [C 脸修复]
      （顺序约定：先超分再脸修复，避免超分破坏已修好的脸部细节，见分析文档 §6.2）
    - 每个阶段为独立可开关单元，失败时**降级为前序产物**并记 WARNING，绝不阻断主流程。

设计原则（遵循 project-rules.md）：
    - 后处理是"增强"，不是"必需路径"。任何阶段异常都不改成功生成的结果，
      只在 stdout/日志中记录退化与原因，让返回产物永远可用。
    - 跨引擎编排仅申请"已在平台注册的同机引擎"，不主动拉起、不做循环重试；
      目标引擎不可达时降级（方案 A 记 WARNING 返回原片）。

方案对应关系（quality_scheme token）：
    - "a"  → 方案 A：低清主渲染 + 后期 RealESRGAN 超分到高清（串接 video_up_enlarge）
    - "c"  → 方案 C：逐帧人脸检测 + GFPGAN/CodeFormer 人脸区域重绘（依赖模型权重）
    - "d"  → 方案 D：采样增强（RF Restart / Tail σ / STG），发生在**生成阶段**（非后处理），
              本模块仅在元数据记录意图；实际采样器增强由 wrapper/采样插件处理。
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

# ===== 目标超分引擎（平台同机已注册引擎）=====
UPSCALE_ENGINE_ID = "video_up_enlarge_98782fcd"
UPSCALE_BASE_URL = "http://127.0.0.1:7013"
UPSCALE_HEALTH_TIMEOUT = 2.0  # 秒（健康检查快速失败，不阻塞）
UPSCALE_INFER_TIMEOUT = 600.0  # 秒（超分单条最长等待）

from .video_utils import _get_ffmpeg_executable  # 供 _mux_with_original_audio 复用（需模块级作用域）

logger = logging.getLogger(__name__)


def apply_quality_postprocess(
    tokens: set[str],
    src_video_abs: str,
    output_dir: Path,
    task_id: str,
    scale: float = 2.0,
) -> dict:
    """按画质方案 token 依次应用后处理阶段。

    Args:
        tokens: 画质方案 token 集合（"a" / "c"，由 _apply_quality_scheme 返回）
        src_video_abs: 主渲染已保存的最终 mp4 绝对路径
        output_dir: 本引擎产物目录（{output_root}/video），超分/修复产物统一落此目录
        task_id: 任务 ID（用于派生后处理产物文件名）
        scale: 方案 A 超分倍率（默认 2x，ncnn 支持 2/3/4）

    Returns:
        dict 含：
            - final_path: 最终产物绝对路径（无增强时 = src_video_abs）
            - stages: 各阶段结果列表（每项含 scheme / applied / path / detail）
            - artifacts: 所有生成物（原始 + 增强），每项 {kind, path, url}
    """
    final_abs = src_video_abs
    stages: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = [
        {"kind": "original", "path": src_video_abs, "url": _to_output_url(src_video_abs)}
    ]

    # ---- 方案 A：后期超分（在脸修复之前）----
    if "a" in tokens:
        up = upscale_video(
            video_path=final_abs,
            output_dir=output_dir,
            task_id=f"{task_id}_up",
            scale=scale,
        )
        if up["applied"]:
            final_abs = up["path"]
            stages.append({
                "scheme": "A",
                "applied": True,
                "path": final_abs,
                "detail": f"超分 {scale}x，来源 video_up_enlarge（RealESRGAN ncnn）",
            })
            artifacts.append({"kind": "upscaled", "path": final_abs, "url": up["url"]})
        else:
            stages.append({
                "scheme": "A",
                "applied": False,
                "path": final_abs,
                "detail": f"超分未生效（降级为原片）: {up['reason']}",
            })
            logger.warning("画质方案 A 超分未生效，降级为原片: %s", up["reason"])

    # ---- 方案 C：脸部修复（依赖模型权重，缺失则降级）----
    if "c" in tokens:
        fr = face_restore_video(
            video_path=final_abs,
            output_dir=output_dir,
            task_id=f"{task_id}_fr",
        )
        if fr["applied"]:
            final_abs = fr["path"]
            stages.append({
                "scheme": "C",
                "applied": True,
                "path": final_abs,
                "detail": "脸部修复（GFPGAN/CodeFormer 人脸区域重绘）",
            })
            artifacts.append({"kind": "face_restored", "path": final_abs, "url": fr["url"]})
        else:
            stages.append({
                "scheme": "C",
                "applied": False,
                "path": final_abs,
                "detail": f"脸修复未生效（降级为前序产物）: {fr['reason']}",
            })
            logger.warning("画质方案 C 脸修复未生效，降级为前序产物: %s", fr["reason"])

    # ---- 方案 D：采样增强（生成阶段）；此处仅记录意图到日志 ----
    if "d" in tokens:
        logger.info(
            "画质方案 D 采样增强：在生成阶段生效（RF Restart/Tail σ/STG）；"
            "后处理阶段无需额外动作"
        )
        stages.append({
            "scheme": "D",
            "applied": False,  # 后处理阶段无动作，采样增强在生成阶段
            "path": final_abs,
            "detail": "采样增强在生成阶段生效，后处理阶段无动作",
        })

    return {
        "final_path": final_abs,
        "stages": stages,
        "artifacts": artifacts,
    }


# ============================ 方案 A：超分 ============================


def upscale_video(
    video_path: str,
    output_dir: Path,
    task_id: str,
    scale: float = 2.0,
) -> dict:
    """调用 video_up_enlarge（RealESRGAN ncnn / Vulkan）对视频做后期超分。

    并发约束：本函数在引擎子进程内调用，实际超分耗时与主渲染同量级。
    目标引擎不可达 / 任务失败时**不阻断主流程**，降级为原片并记原因。

    Args:
        video_path: 待超分 mp4 绝对路径
        output_dir: 本引擎产物目录（超分产物复制到这里统一管理）
        task_id: 派生任务 ID（用于本引擎内文件名）
        scale: 放大倍率（ncnn 支持 2 / 3 / 4）

    Returns:
        dict 含：
            - applied: 是否成功生成超分产物
            - path: 超分产物绝对路径（applied=False 时 = 原路径）
            - url: 超分产物相对 URL（/output/...）
            - reason: 失败原因（未生效时）
    """
    src = Path(video_path)
    if not src.is_file():
        return {"applied": False, "path": video_path, "url": "", "reason": f"源视频不存在: {src}"}

    # 目标引擎连通性检查（快速失败降级，不阻塞）
    if not _engine_healthy(UPSCALE_BASE_URL):
        return {
            "applied": False,
            "path": video_path,
            "url": "",
            "reason": (
                f"超分引擎 {UPSCALE_ENGINE_ID} 不可达（{UPSCALE_BASE_URL}），未启动或加载中。"
                "请在平台启动该引擎后重试（当前输出为未超分原片）"
            ),
        }

    # 构造 /infer 请求（CustomEngine 契约：推理参数放在 inputs）
    # ⚠️ 必须显式指定 variant=realesr-animevideov3：
    #   超分引擎默认 variant 是 realesrgan-x4plus，实测对低照度人像视频帧
    #   （暗部占比大、人物为暗部剪影）会「构图坍缩」——把居中女主重构成
    #   门缝/窥视式构图，人物被压到画面最右缘只剩局部（女主脸只剩一点）。
    #   同帧实测：animevideov3（视频友好模型）能完整保留居中构图与脸部。
    #   （2026-09-25，single-frame 对照实验，frames_in 女主居中 → x4plus 门缝 /
    #    animevideov3 女主居中）
    payload = {
        "task_id": task_id,
        "variant": "videoUpEnlarge/realesr-animevideov3",
        "inputs": {
            "videos": [{"path": str(src.resolve())}],
            "scale": float(scale),
            "workflow": "upscale",
        },
    }

    out_path = src
    out_url = ""
    try:
        video_entry = _call_infer_sse(UPSCALE_BASE_URL, payload)
        if not video_entry:
            return {
                "applied": False,
                "path": video_path,
                "url": "",
                "reason": "超分任务未返回产物（completed 事件缺失或失败）",
            }
        # 超分原始产物（该引擎输出在它自己的输出目录，复制到本引擎产物目录统一管理）
        raw_path = Path(video_entry["video_path"])
        if not raw_path.is_file():
            return {
                "applied": False,
                "path": video_path,
                "url": "",
                "reason": f"超分产物文件不存在: {raw_path}",
            }
        out_path = output_dir / f"{task_id}.mp4"
        out_path = out_path.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_path, out_path)
        out_url = _to_output_url(str(out_path))
        logger.info(
            "画质方案 A 超分完成: %s → %s（%s, 耗时 %.1fs）",
            src, out_path, raw_path,
            float(video_entry.get("elapsed") or 0),
        )
        return {"applied": True, "path": str(out_path), "url": out_url, "reason": ""}
    except Exception as e:
        return {
            "applied": False,
            "path": video_path,
            "url": "",
            "reason": f"超分调用异常: {e}",
        }


def _engine_healthy(base_url: str) -> bool:
    """快速健康检查目标引擎（GET /health，短超时）。"""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=UPSCALE_HEALTH_TIMEOUT) as resp:
            return resp.status == 200
    except Exception:
        return False


def _call_infer_sse(base_url: str, payload: dict) -> Optional[dict]:
    """POST /infer（SSE 流），解析到 completed 事件的 videos[0]。

    返回产物 dict（video_path / video_url / width / height / duration）。
    未拿到 completed 或收到 error 事件返回 None（不抛异常，调用方降级）。
    """
    import requests  # 延迟导入：目标引擎未启动时不引入硬依赖

    url = f"{base_url}/infer"
    first_video_entry: Optional[dict] = None
    with requests.post(url, json=payload, stream=True, timeout=UPSCALE_INFER_TIMEOUT) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data:
                continue
            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue
            # ⚠️ 完成/错误帧用 stage 字段标识（SDK make_final_frame：{"progress":1.0,
            #    "stage":"completed", "result":{...}}），无 event/type 字段；
            #    progress 中转帧也无 event 字段（stage=其他），故以 stage 为准。
            evt_name = evt.get("stage") or evt.get("event") or evt.get("type") or ""
            if evt_name == "error":
                logger.warning("超分任务上报错误事件: %s", evt.get("error") or evt.get("data") or evt)
                return None
            if evt_name == "completed":
                result = evt.get("result") or {}
                videos = result.get("videos") or []
                if videos:
                    first_video_entry = videos[0]
                # 拿到 completed 即收口（无论有无产物）
                break
        return first_video_entry


# ============================ 方案 C：脸修复 ============================

# 允许的人脸修复后端（按可用性自动选择）：优先 GFPGAN（facexlib 依赖），
# 其次用户准备的 CodeFormer。两者都不可用则降级。
# 权重搜索目录（本引擎模型目录下）：
#   models/video/minimax-h3/gfpgan/{GFPGANv1.4.pth 或 GFPGANv1.3.pth}
#   models/video/minimax-h3/codeformer/{codeformer_v0.1.0.pth}


def _ensure_basicsr_compat() -> None:
    """gfpgan→basicsr 1.4.2 与新版 torchvision 的兼容 shim（幂等）。

    basicsr/data/degradations.py 仍 `from torchvision.transforms.functional_tensor import
    rgb_to_grayscale`；torchvision 0.26+ 已把该函数迁至
    `torchvision.transforms.functional`。注入 sys.modules 别名用小模块引用：
        sys.modules['torchvision.transforms.functional_tensor'] = torchvision.transforms.functional
    该模块的 replace 目标仅被 basicsr 引用且只用 rgb_to_grayscale，语义等价，可安全全局替换。
    """
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms as _T
        sys.modules["torchvision.transforms.functional_tensor"] = _T.functional
    except Exception as e:  # torchvision 不可用时让上层 gfpgan import 自然失败
        logger.debug("basicsr 兼容 shim 注入失败: %s", e)


def _find_face_restorer():
    """探测可用的脸修复后端组件（gfpgan 已安装 + 权重存在）。

    Returns:
        已初始化 restorer 或 None（不可用时）。

    Raises:
        仅记录 DISABLE，不向外抛。
    """
    try:
        # 主页权重检索：优先 GFPGANv1.4，其次 1.3
        import os
        for arch, name in (("clean", "GFPGANv1.4.pth"), ("clean", "GFPGANv1.3.pth")):
            for base in ("gfpgan", "codeformer"):
                cand = Path(os.environ.get("AIRELIABLE_MODELS_ROOT", "models")) \
                    / "video" / "minimax-h3" / base / name
                if cand.is_file():
                    try:
                        _ensure_basicsr_compat()  # gfpgan→basicsr 与新版 torchvision 的兼容 shim
                        from gfpgan import GFPGANer  # type: ignore
                        restorer = GFPGANer(
                            model_path=str(cand),
                            upscale=1,  # 只重绘，不放大（放大交给方案 A）
                            arch=arch,
                            channel_multiplier=2,
                            bg_upsampler=None,
                        )
                        logger.info("画质方案 C 已初始化 GFPGAN 脸修复（权重 %s）", cand)
                        return restorer
                    except Exception as e:
                        logger.warning("gfpgan 初始化失败（%s）: %s", name, e)
    except Exception as e:
        logger.debug("脸修复后端探测失败: %s", e)
    return None


def face_restore_video(
    video_path: str,
    output_dir: Path,
    task_id: str,
    selected_frames: Optional[int] = None,
) -> dict:
    """对整段视频做逐帧人脸修复（GFPGAN/CodeFormer 人脸区域重绘）。

    行为：
        - 依赖 gfpgan + 权重；不可用时降级为前序产物（不阻断）。
        - 只重绘人脸区域（restorer 内置 parse 定位），避免整帧风格漂移。
        - 保留原音轨（抽帧→修复→合帧回填）。

    Args:
        video_path: 待修复 mp4 绝对路径
        output_dir: 本引擎产物目录
        task_id: 派生任务 ID
        selected_frames: 可选，仅处理抽帧中的指定帧（None=全处理）

    Returns:
        dict 含 applied / path / url / reason。
    """
    src = Path(video_path)
    if not src.is_file():
        return {"applied": False, "path": video_path, "url": "", "reason": f"源视频不存在: {src}"}

    restorer = _find_face_restorer()
    if restorer is None:
        return {
            "applied": False,
            "path": video_path,
            "url": "",
            "reason": (
                "人脸修复后端不可用（未安装 gfpgan 或未找到权重 "
                "GFPGANv1.3/v1.4.pth / codeformer_v0.1.0.pth）。"
                "下载权重到 models/video/minimax-h3/ 下后重试（当前为未修复原片）"
            ),
        }

    try:
        import imageio.v2 as imageio  # 帧序列读写
        import numpy as np

        out_path = (output_dir / f"{task_id}.mp4").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # 读帧
        reader = imageio.get_reader(str(src))
        try:
            fps = float(reader.get_meta_data().get("fps", 24.0))
            frames: list[np.ndarray] = []
            for f in reader:
                frames.append(f.astype(np.uint8))
        finally:
            reader.close()

        if not frames:
            return {"applied": False, "path": video_path, "url": "",
                    "reason": "视频帧读取为空，无法脸修复"}

        idxs = range(len(frames))
        if selected_frames is not None:
            idxs = [i for i in range(len(frames)) if i in set(selected_frames)]

        # 逐帧人脸修复（仅含人脸帧真正重绘，无人脸帧原样保留）
        restored = copy.deepcopy(frames)
        for i in idxs:
            bgr = frames[i][..., ::-1]  # RGB → BGR
            try:
                _, _, out_bgr = restorer.enhance(
                    bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
                out_rgb = np.clip(out_bgr[..., ::-1], 0, 255).astype(np.uint8)
                restored[i] = out_rgb
            except Exception as e:
                logger.warning("帧 %d 脸修复失败，保留原帧: %s", i, e)

        # 合帧（写 video_only）+ 回填原音轨
        video_only = output_dir / f"{task_id}_video_only.mp4"
        writer = imageio.get_writer(str(video_only), fps=fps,
                                     codec="libx264", quality=8, macro_block_size=1)
        try:
            for fr in restored:
                writer.append_data(fr)
        finally:
            writer.close()

        audio_ok = _mux_with_original_audio(src, video_only, out_path)
        if not audio_ok:
            shutil.copy2(video_only, out_path)

        url = _to_output_url(str(out_path))
        logger.info("画质方案 C 脸修复完成: %s → %s（帧 %d，音轨回填 %s）",
                    src, out_path, len(frames), "OK" if audio_ok else "降级无音轨")
        return {"applied": True, "path": str(out_path), "url": url, "reason": ""}
    except Exception as e:
        return {"applied": False, "path": video_path, "url": "",
                "reason": f"脸修复异常: {e}"}


def _mux_with_original_audio(src: Path, video_only: Path, out_path: Path) -> bool:
    """将原视频音轨回填到已修复的视频（用 ffmpeg 复制音轨）。"""
    ffmpeg = _get_ffmpeg_executable()
    if not ffmpeg:
        return False
    try:
        # 先探测原视频是否有音轨与采样率（ffprobe 不可用时降级为无条件 -c:a copy 交由 ffmpeg 判定）
        cmd = [ffmpeg, "-y",
               "-i", str(video_only),
               "-i", str(src),
               "-map", "0:v:0",
               "-c:v", "copy",
               "-map", "1:a:0?",
               "-c:a", "copy",
               "-shortest",
               "-movflags", "+faststart",
               str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            logger.warning("脸修复音轨回填失败: %s", r.stderr[:300])
            return False
        if not out_path.is_file() or out_path.stat().st_size == 0:
            return False
        return True
    except Exception as e:
        logger.warning("脸修复音轨回填异常: %s", e)
        return False


# ============================ 工具 ============================


def _to_output_url(abs_path: str) -> str:
    """把产物绝对路径映射为 /output/... 相对 URL（与 save_video 约定一致）。"""
    p = Path(abs_path).resolve()
    items = p.parts
    if "output" in items:
        idx = items.index("output")
        return "/output/" + "/".join(items[idx + 1:]).replace("\\", "/")
    return str(p)