"""MiniMax-H3 视频生成引擎 HTTP 服务（FastAPI）。

8 个标准端点 / CORS / 日志端点 / 静态挂载 / 根路径跳转由 SDK 骨架
（``build_engine_app``）提供；本文件保留 ``/load``（需读原始 body 的
``acceleration_override``）、方案管理端点、``/recommended`` 与 ``/upload``。

启动：
    uv run --no-sync --project envs/video-minimax-h3 \\
        python -m engines.minimax_h3_video_a1b2c3.server --port 7006

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用模型变体）
    POST /load           加载 MiniMax-H3 模型权重（支持 acceleration_override）
    POST /unload         卸载模型（释放显存）
    POST /infer          视频生成推理（SSE 流式进度）
    GET  /status         状态查询
    POST /lora/load      加载 LoRA
    POST /lora/unload    卸载 LoRA
    GET  /output/*       静态视频文件（推理产物）
    GET  /log_level      查询/调整日志级别（v0.2.5+ T6.2）
    POST /upload         参考图片/视频上传（WebUI I2V/FL2V/Ref2V/编辑模式，返回引擎侧路径）
    GET  /               307 重定向到 /ui/
    GET  /ui/*           原生 WebUI 静态文件（webui/ 三件套，FLUX 风格）

WebUI 说明（引擎 WebUI 风格统一）：
    - Gradio UI（ui.py）已废弃，改为原生 HTML/CSS/JS 三件套（webui/）
    - 风格基底与 FLUX 引擎一致（ENGINE_WEBUI_FLUX_STYLE_UNIFY 分析）
    - 五模式（t2v/i2v/first_last/fl2v/ref2v/video_edit）参考图/视频经
      ``POST /upload`` 落盘后以 ``images[].path`` / ``videos[].path``
      传入 ``/infer``
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request

from aireliable_engine import LoadRequest, build_engine_app, mount_upload, run_engine_server

from .config import EngineConfig, RECOMMENDED_STEPS, RECOMMENDED_TABLE, parse_args
from .engine import MiniMaxH3Engine

logger = logging.getLogger(__name__)

_WEBUI_DIR = Path(__file__).parent / "webui"


# ===== 良性连接复位噪音压制 =====
# Windows asyncio Proactor 在被父进程强制终止时，_ProactorBasePipeTransport.
# _call_connection_lost 会对已关闭的 socket 再调用 shutdown(SHUT_RDWR)，抛出
# ConnectionResetError [WinError 10054]，表现为控制台/日志里一堆
# "Exception in callback ... _call_connection_lost(None)" ERROR。
# 属进程退出时的良性噪音（任务已 succeeded/failed 落库），不吞会刷屏干扰排查。

_SILENCE_CONNECTION_RESET = {"ConnectionResetError"}


def _install_loop_exception_handler() -> None:
    """给当前事件循环装自定义 exception handler，静默良性连接复位噪音。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # 非事件循环上下文（如单测），跳过

    orig_handler = loop.get_exception_handler()

    def _handler(_loop, context) -> None:
        # 外部调用方也 set_exception_handler，需与原始 handler 组合
        exc = context.get("exception")
        if exc is not None:
            exc_name = exc.__class__.__name__
            if exc_name in _SILENCE_CONNECTION_RESET:
                # socket.shutdown 抛 ConnectionResetError：进程退出断管，忽略
                return
        if orig_handler is not None:
            orig_handler(_loop, context)
        else:
            _loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


# ===== 全局引擎实例 =====
# 由 SDK 骨架的 startup_hook 绑定，供本文件的路由使用
_engine: Optional[MiniMaxH3Engine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> MiniMaxH3Engine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 钩子（由 SDK 骨架调用）=====

async def _on_startup(engine: MiniMaxH3Engine) -> None:
    """启动钩子：噪声压制 → 绑定全局实例 → 异步刷新 variant 扫描。"""
    global _engine

    # 压制被强杀时的 ConnectionResetError [WinError 10054] 良性噪音
    _install_loop_exception_handler()

    _engine = engine
    # v0.2.11+ 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    # __init__ 阶段的同步 _refresh_scan 只是 fallback，此处用平台 API 重新检测
    await engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, loras=%d",
        engine.state_machine.state.value,
        len(engine._available_variants),
        len(engine._available_loras),
    )


async def _on_shutdown(engine: MiniMaxH3Engine) -> None:
    """关闭钩子：卸载模型并清空全局引用。"""
    global _engine, _config

    logger.info("===== MiniMax-H3 视频生成引擎关闭 =====")
    try:
        if engine.wrapper.is_loaded:
            try:
                await engine._unload_impl()
            except Exception as e:
                logger.warning("关闭时卸载模型失败: %s", e)
    finally:
        _engine = None
        _config = None


# ===== 引擎特有路由 =====

router = APIRouter()


@router.post("/load")
async def load(request: LoadRequest, raw_request: Request) -> dict:
    """加载 MiniMax-H3 模型权重。

    v0.6+：支持通过 request body 的 inputs.acceleration_override 字段
    临时覆盖当前方案的加速配置（高级模式用，不持久化）。
    """
    try:
        # v0.6+：从原始 body 提取 acceleration_override（LoadRequest schema 不含此字段）
        acceleration_override = None
        try:
            raw_body = await raw_request.json()
            inputs = raw_body.get("inputs", {}) if isinstance(raw_body, dict) else {}
            acceleration_override = inputs.get("acceleration_override") if isinstance(inputs, dict) else None
            if acceleration_override:
                logger.info("检测到 acceleration_override: %s", acceleration_override)
        except Exception:
            pass  # body 不是 JSON 或解析失败，忽略

        response = await get_engine().load(request, acceleration_override=acceleration_override)
        return response.model_dump()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("加载失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schemes")
async def list_schemes() -> dict:
    """v0.6+：列出所有可用方案（A/B/C/D）。

    返回方案列表，每项含 id / name / description / vram_required / precision。
    """
    engine = get_engine()
    return {
        "schemes": engine.list_schemes(),
        "default_scheme_id": engine.get_default_scheme(),
        "loaded_scheme_id": engine.wrapper.loaded_scheme_id,
    }


@router.get("/recommended")
async def recommended_steps() -> dict:
    """返回各生成方式（workflow）的推荐最佳采样步数表（含模型归属/中文方式/理由）。

    - recommended_steps: mode → steps 映射，webui 切换工作流据此自动跳到推荐步数；
      引擎 generate 未显式传 num_inference_steps 时也用此兜底。
    - recommended_table: 完整参考表（model/label/steps/reason），供 webui「推荐参考表」
      弹窗展示。数据源为 config.RECOMMENDED_TABLE（单一来源，改一处全同步）。
    """
    return {
        "recommended_steps": RECOMMENDED_STEPS,
        "recommended_table": RECOMMENDED_TABLE,
    }


@router.get("/schemes/default")
async def get_default_scheme() -> dict:
    """v0.6+：获取当前默认方案 ID。"""
    engine = get_engine()
    return {
        "default_scheme_id": engine.get_default_scheme(),
        "loaded_scheme_id": engine.wrapper.loaded_scheme_id,
    }


@router.post("/schemes/default")
async def set_default_scheme(body: dict) -> dict:
    """v0.6+：设置默认方案并持久化。

    请求体：
        {"scheme_id": "A"}

    设置后下次加载 SDNQ variant 时将自动使用该方案。
    已加载的模型不会立即切换（需 unload + load 新方案）。
    """
    engine = get_engine()
    scheme_id = body.get("scheme_id")
    if not scheme_id:
        raise HTTPException(status_code=400, detail="缺少 scheme_id 参数")

    ok = engine.set_default_scheme(scheme_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"方案 {scheme_id} 不存在（可用: A / B / C / D）",
        )
    return {
        "ok": True,
        "default_scheme_id": scheme_id,
        "message": f"默认方案已设置为 {scheme_id}，下次加载 SDNQ variant 时生效",
    }


mount_upload(
    router,
    lambda: get_config().output_root / "uploads",
    log_label="参考文件",
)


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config

    if config is None:
        config = parse_args()
    _config = config

    return build_engine_app(
        MiniMaxH3Engine,
        config=config,
        title="MiniMax-H3 Video Engine",
        description="AIReliable 平台的 MiniMax-H3 视频生成引擎（FL2VA / Ref2VA 双范式，含音频 VAE）",
        version="0.1.0",
        static_dirs={"/ui": _WEBUI_DIR, "/output": config.output_root},
        extra_routers=[router],
        skip_endpoints=["/load"],  # /load 需读原始 body 的 acceleration_override
        startup_hook=_on_startup,
        shutdown_hook=_on_shutdown,
    )


# ===== 入口 =====

def main() -> None:
    """命令行启动入口。"""
    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


if __name__ == "__main__":
    main()
