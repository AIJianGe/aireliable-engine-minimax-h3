"""MiniMax-H3 视频生成引擎推理核心。

继承 ``CustomEngine``（全自定义模式 / integrated 范式），实现 7 个
``_xxx_impl`` 方法，封装 MiniMaxH3Wrapper 的加载/推理/卸载流程。

8 端点契约：
    - /capabilities    返回扫描到的 MiniMax-H3 模型变体
    - /load            调用 MiniMaxH3Wrapper.load() 加载模型权重
    - /unload          调用 MiniMaxH3Wrapper.unload() 释放显存
    - /infer           解析 VideoInputs → 确定生成模式 → 调用 generate → 保存视频 → SSE 推送
    - /status          返回引擎状态（已加载变体 / 设备 / 显存 / offload 级别）
    - /lora/load       代理到 LoRAManager.load_lora
    - /lora/unload     代理到 LoRAManager.unload_lora
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    EngineBusyError,
    InferRequest,
    LoadRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
)
from aireliable_engine.state_machine import RuntimeState

from .config import EngineConfig
from .inference_engine import InferenceEngine
from .minimax_h3_wrapper import MiniMaxH3Wrapper
from .model_scanner import ModelScanner
from .progress_reporter import ProgressReporter
from .prompt_processor import PromptProcessor
from .scheme_manager import SchemeConfig, SchemeManager
from .scheme_state_store import SchemeStateStore
from .scheme_validator import SchemeValidator
from .schemas import VideoInputs, VideoVariantInfo
from .video_utils import save_video
from . import quality_postprocess  # 45-专项：画质方案后处理编排

logger = logging.getLogger(__name__)


class MiniMaxH3Engine(CustomEngine):
    """MiniMax-H3 视频生成引擎。

    全自定义模式（integrated 范式），直接控制模型加载/推理流程，
    不走工作流节点系统。支持 FL2VA / Ref2VA 双范式视频生成。
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config

        # v0.6+ 方案管理组件
        # schemes/ 目录与 engine.py 同级
        schemes_dir = Path(__file__).parent / "schemes"
        self.scheme_manager = SchemeManager(schemes_dir=schemes_dir, logger=logger)
        # 方案状态持久化（存储用户选择的默认方案）
        self.scheme_store = SchemeStateStore(
            data_root=config.data_root,
            engine_id=config.engine_id,
            logger=logger,
        )
        # 方案校验器（切换方案前校验合法性）
        self.scheme_validator = SchemeValidator(
            scheme_manager=self.scheme_manager,
            models_root=config.models_root,
            logger=logger,
        )

        # 核心组件
        self.wrapper = MiniMaxH3Wrapper(
            models_root=config.models_root,
            model_dir=config.model_dir,
            logger=logger,
        )
        # v0.6+ 推理引擎（从 wrapper 分离的推理逻辑）
        self.inference_engine = InferenceEngine(
            wrapper=self.wrapper,
            logger=logger,
        )
        # v0.2.11+ 方案 B：注入 engine=self，启用平台 API 声明式检测
        self.scanner = ModelScanner(
            models_root=config.models_root,
            model_dir=config.model_dir,
            engine=self,
            logger=logger,
        )

        # 提示词处理器（Context-IR 简化版）
        self.prompt_processor = PromptProcessor(logger=logger)

        # 启动时扫描模型目录（同步初始化为空，异步扫描在 lifespan 中触发）
        self._available_variants: list = []
        self._refresh_scan()

        # LoRA 列表（启动时扫描）
        self._available_loras: list[str] = self.scanner.scan_loras()

        # 输出根目录（供 BaseEngine.get_output_path 使用）
        self.output_root = config.output_root

        # 读取 manifest 中的 default_variant（供 BaseEngine.infer auto-load 使用）
        # 优先级：request.variant > self.default_variant > 第一个 available variant
        self.default_variant = self._read_default_variant_from_manifest(
            Path(__file__).parent / "manifest.yaml"
        )

        # v0.6+ 加载默认方案（从 state store 读取，下次启动自动使用）
        self._default_scheme_id: Optional[str] = self.scheme_store.get_default_scheme()
        if self._default_scheme_id:
            logger.info("从 state store 加载默认方案: %s", self._default_scheme_id)
        else:
            # 启动时按显存自动选档并持久化
            auto_scheme = self.scheme_manager.auto_select_scheme()
            if auto_scheme:
                self._default_scheme_id = auto_scheme.scheme
                self.scheme_store.set_default_scheme(auto_scheme.scheme)
                logger.info(
                    "自动选档并持久化默认方案: %s（%s）",
                    auto_scheme.scheme, auto_scheme.name,
                )

    # ============ 扫描 ============

    def _refresh_scan(self) -> None:
        """刷新模型扫描结果（同步 fallback，用于 __init__ 阶段）。

        v0.2.11+ 方案 B：此方法走 fallback 本地扫描逻辑。
        平台 API 异步扫描请调用 `_refresh_scan_async()`（lifespan startup 阶段调用）。
        """
        logger.info("刷新模型扫描（本地 fallback）: %s", self.config.model_dir)
        self._available_variants = self.scanner.scan_variants()
        # 同步刷新 LoRA 列表
        self._available_loras = self.scanner.scan_loras()

    async def _refresh_scan_async(self) -> None:
        """v0.2.11+ 方案 B：异步刷新模型扫描结果（优先平台 API）。

        在 lifespan startup 阶段调用，此时平台 API 已就绪。
        平台不可达时 fallback 到本地扫描。
        """
        logger.info("异步刷新模型扫描（方案 B 平台 API）: %s", self.config.model_dir)
        try:
            self._available_variants = await self.scanner.scan_variants_async()
        except Exception as e:
            logger.warning("异步扫描失败，回退到本地扫描: %s", e)
            self._available_variants = self.scanner.scan_variants()
        # 刷新 LoRA 列表
        self._available_loras = self.scanner.scan_loras()

    # ============ 7 个 _xxx_impl 实现 ============

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities：返回可用模型变体。"""
        variants: list[VariantInfo] = []
        for v in self._available_variants:
            variants.append(VideoVariantInfo(
                name=v.name,
                base_model=v.base_model,
                precision=v.precision,
                loader=v.loader,
                vram_required=v.vram_required,
                available=v.available,
                resolution=v.resolution,
                fps=v.fps,
                duration=v.duration,
                max_duration=v.max_duration,
                supports_audio=v.supports_audio,
            ))

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=self._available_loras,
            supported_sizes=["768x432", "1280x720"],  # 支持的输出分辨率
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema（MCP 专属层动态 tool 数据源）
            input_schema=VideoInputs.model_json_schema(),
        )

    async def load(
        self,
        request: LoadRequest,
        acceleration_override: Optional[dict] = None,
    ) -> LoadResponse:
        """POST /load - 非阻塞 fire-and-forget 模式（MiniMax-H3 专用）。

        MiniMax-H3 模型很大（60GB+ 权重），同步加载需要 1-3 分钟。
        传统同步 /load 会导致：
            1. HTTP 请求长时间挂起（容易超时）
            2. asyncio.to_thread 的工作线程持有 GIL，阻塞事件循环
            3. /health 端点无法响应 → 平台健康检查失败 → 引擎被 kill 重启

        采用 fire-and-forget 架构：
            1. /load 收到请求 → 立即转换状态到 LOADING → 返回 state="loading"
            2. 实际加载在后台 asyncio task 中执行（不阻塞 /load HTTP 响应）
            3. 加载完成 → 状态转为 READY；失败 → 状态转为 ERROR
            4. /health 可以正常响应（即使 GIL 被后台线程持有，事件循环仍能
               在 GIL 释放的间隙处理 /health 的轻量级请求）

        调用方（API/WebUI）应在收到 state="loading" 后轮询 /status 直到
        state="ready" 或 "error"。

        兼容性：保留 _op_lock 并发防护，但在 LOADING 状态下允许 /health
        和 /status 查询。

        v0.6+：
            - acceleration_override 参数：来自 WebUI 高级模式，
              允许用户临时覆盖当前方案的加速配置（不持久化）。
              传入后会覆盖 scheme.acceleration / block_swap 等配置。
        """
        async with self._op_lock:
            if self.state_machine.is_busy():
                raise EngineBusyError(
                    f"引擎忙（state={self.state_machine.state.value}），请稍后再试"
                )
            if self.state_machine.state == RuntimeState.INFERRING:
                raise EngineBusyError("推理进行中，无法加载")

            # v0.3.7+ 修复：READY/ERROR 状态时先清理再 load
            current = self.state_machine.state
            if current == RuntimeState.READY:
                logger.info(
                    "load()：当前状态 READY，先 unload 释放资源再加载 variant=%s",
                    request.variant,
                )
                try:
                    await self._unload_impl()
                    self.state_machine.transition(RuntimeState.IDLE)
                except Exception as e:
                    logger.warning("load()：unload 失败，强制 reset 状态机: %s", e)
                    try:
                        self.state_machine.transition(RuntimeState.ERROR)
                        self.state_machine.reset()
                    except Exception:
                        pass
            elif current == RuntimeState.ERROR:
                logger.info("load()：当前状态 ERROR，reset 到 IDLE 再加载")
                try:
                    self.state_machine.reset()
                except Exception as e:
                    logger.warning("load()：reset 失败: %s", e)

            # 转换状态到 LOADING
            self.state_machine.transition(RuntimeState.LOADING)

        # 在 _op_lock 外启动后台加载任务（避免长时间持锁阻塞其他操作）
        # 注意：此时 _op_lock 已释放，但 state_machine 已是 LOADING，
        # 其他 load/unload 调用会因 is_busy() 被拒绝
        variant = request.variant or self.default_variant
        if not variant:
            # 没有 variant 参数，也没有 default_variant，转 ERROR
            self.state_machine.transition(RuntimeState.ERROR)
            raise RuntimeError(
                "未指定 variant 且引擎未配置 default_variant，无法加载"
            )

        # v0.6.22+ 修复 TC-1.3.4：同步预校验 variant 是否存在。
        # 原本此校验在 _load_in_background 中执行，但因为是 fire-and-forget 后台任务，
        # FileNotFoundError 不会同步传到 /load HTTP 响应，导致不存在的 variant 也返回 200。
        # 现在在同步部分预先校验，不存在则抛 FileNotFoundError（被 server.py 捕获返回 404）。
        variant_path = self.scanner.resolve_variant_path(variant)
        if variant_path is None:
            # 回滚状态机：LOADING → IDLE（因为还没真正开始加载）
            self.state_machine.transition(RuntimeState.IDLE)
            raise FileNotFoundError(
                f"variant 不存在: {variant} "
                f"（可用: {[v.name for v in self._available_variants]}）"
            )

        logger.info(
            "load()：已启动后台加载任务 variant=%s, acceleration_override=%s，/load 立即返回 state=loading",
            variant, "有" if acceleration_override else "无",
        )

        # 创建后台任务（不 await，fire-and-forget）
        # v0.6+：传 acceleration_override 给后台任务
        asyncio.create_task(
            self._load_in_background(variant, acceleration_override=acceleration_override)
        )

        # 立即返回 state="loading"
        return LoadResponse(
            variant=variant,
            state="loading",
            vram_used="0MB",
            load_time=0.0,
        )

    async def _load_in_background(
        self,
        variant: str,
        acceleration_override: Optional[dict] = None,
    ) -> None:
        """后台加载模型权重（fire-and-forget 模式的实际加载逻辑，v0.6+ 多方案版）。

        v0.6+ 改造：
            - 从 variant 名反查适用方案（SDNQ variant → 方案 A/B/C，BF16 variant → 方案 D）
            - 优先使用 state store 持久化的默认方案（针对 SDNQ 选 A/B/C 中的哪个）
            - 调用 SchemeValidator 校验方案合法性（显存 + 模型文件）
            - 将 SchemeConfig 传给 wrapper.load()
            - acceleration_override 非空时，覆盖 scheme 的 acceleration / block_swap 配置

        加载完成后更新 state_machine：
            - 成功 → READY
            - 失败 → ERROR

        注意：此任务在后台运行，不阻塞 /load 的 HTTP 响应。
        """
        load_start_time = time.time()
        try:
            logger.info(
                "后台加载任务开始: variant=%s, acceleration_override=%s",
                variant, "有" if acceleration_override else "无",
            )

            variant_path = self.scanner.resolve_variant_path(variant)
            if variant_path is None:
                raise FileNotFoundError(
                    f"variant 不存在: {variant} "
                    f"（可用: {[v.name for v in self._available_variants]}）"
                )

            # v0.6+：解析方案配置
            scheme = self._resolve_scheme_for_variant(variant)
            if scheme is None:
                raise RuntimeError(
                    f"无法为 variant={variant} 解析方案配置，"
                    f"请检查 schemes/ 目录下的 yaml 文件"
                )

            # v0.6+：若传入 acceleration_override，覆盖 scheme 的加速配置
            if acceleration_override:
                scheme = self._apply_acceleration_override(scheme, acceleration_override)
                logger.info(
                    "已应用 acceleration_override: scheme=%s 的加速配置已被覆盖",
                    scheme.scheme,
                )

            # v0.6+：方案合法性校验（显存 + 模型文件，仅警告不阻断）
            validation = self.scheme_validator.validate(
                scheme_id=scheme.scheme,
                variant_name=variant,
            )
            if not validation.ok:
                # 硬错误（方案不存在 / 方案与 variant 不匹配 / 模型目录缺失）→ 抛出
                raise RuntimeError(
                    f"方案校验失败: {validation.errors}"
                )
            if validation.warnings:
                # 警告（显存不足 / 标志文件缺失）→ 记录但继续
                for w in validation.warnings:
                    logger.warning("方案校验警告: %s", w)

            logger.info(
                "开始加载 variant: %s (path=%s, scheme=%s, precision=%s, offload=%s)",
                variant, variant_path, scheme.scheme, scheme.precision, scheme.offload_level,
            )

            # 在线程池中执行同步加载（不阻塞事件循环）
            # v0.6+：传 scheme 而非 offload_level
            await asyncio.to_thread(
                self.wrapper.load,
                variant,
                str(variant_path),
                scheme,
                self.config.device,
            )

            load_time = time.time() - load_start_time
            vram_used = self.wrapper.get_vram_used()

            logger.info(
                "后台加载完成: variant=%s, scheme=%s, device=%s, offload=%s, "
                "load_time=%.2fs, vram_used=%.1fMB",
                variant, self.wrapper.loaded_scheme_id, self.wrapper.device,
                self.wrapper.offload_level, load_time, vram_used,
            )

            # 记录方案使用历史
            self.scheme_store.record_history(
                scheme=scheme.scheme, variant=variant, success=True,
            )

            # 加载成功，转换状态到 READY
            # 若加载期间有并发推理请求将状态改为 ERROR，需先 reset 回 IDLE 再转到 READY
            if self.state_machine.state == RuntimeState.ERROR:
                logger.warning("加载期间状态被并发请求改为 ERROR，reset 后恢复到 READY")
                self.state_machine.reset()  # ERROR → IDLE
                self.state_machine.transition(RuntimeState.LOADING)  # IDLE → LOADING
            self.state_machine.transition(RuntimeState.READY)
            logger.info(
                "状态已转换到 READY: variant=%s, scheme=%s, vram_used=%.1fMB",
                variant, scheme.scheme, vram_used,
            )

        except Exception as e:
            logger.exception("后台加载失败: variant=%s", variant)
            # 记录失败历史
            try:
                self.scheme_store.record_history(
                    scheme=self._default_scheme_id or "?",
                    variant=variant, success=False, error=str(e),
                )
            except Exception:
                pass
            # 加载失败，转换状态到 ERROR
            try:
                self.state_machine.transition(RuntimeState.ERROR)
            except Exception:
                # 状态转换失败，强制 reset
                try:
                    self.state_machine.reset()
                except Exception:
                    pass
            logger.error(
                "后台加载失败，状态已转换到 ERROR: variant=%s, error=%s",
                variant, e,
            )

    def _apply_acceleration_override(
        self,
        scheme: SchemeConfig,
        override: dict,
    ) -> SchemeConfig:
        """v0.6+：将 acceleration_override 应用到 SchemeConfig，返回新的 SchemeConfig。

        override 结构：
            {
                "sage_attention": {"enabled": True/False},
                "lora": {"enabled": True/False, "scale": 1.0} or None,
                "first_block_cache": {"enabled": True/False},
                "block_swap": {"enabled": True/False},
            }

        Args:
            scheme: 原始方案配置
            override: 加速方案覆盖配置

        Returns:
            新的 SchemeConfig（深拷贝 + 覆盖），不影响原始 scheme
        """
        import copy
        new_scheme = copy.deepcopy(scheme)

        # SageAttention
        sage_override = override.get("sage_attention") or {}
        if sage_override:
            new_scheme.acceleration.sage_attention.enabled = bool(
                sage_override.get("enabled", new_scheme.acceleration.sage_attention.enabled)
            )

        # LoRA
        lora_override = override.get("lora")
        if lora_override is None:
            # 用户禁用 LoRA
            if new_scheme.acceleration.lora is not None:
                new_scheme.acceleration.lora.enabled = False
        else:
            # 用户启用或调整 LoRA
            from .scheme_manager import LoRAConfig
            if new_scheme.acceleration.lora is None:
                new_scheme.acceleration.lora = LoRAConfig(enabled=True)
            new_scheme.acceleration.lora.enabled = bool(
                lora_override.get("enabled", new_scheme.acceleration.lora.enabled)
            )
            if "scale" in lora_override:
                new_scheme.acceleration.lora.scale = float(lora_override["scale"])

        # FirstBlockCache
        fbc_override = override.get("first_block_cache") or {}
        if fbc_override:
            from .scheme_manager import FirstBlockCacheConfig
            if new_scheme.acceleration.first_block_cache is None:
                new_scheme.acceleration.first_block_cache = FirstBlockCacheConfig()
            new_scheme.acceleration.first_block_cache.enabled = bool(
                fbc_override.get("enabled", new_scheme.acceleration.first_block_cache.enabled)
            )

        # BlockSwap
        bs_override = override.get("block_swap") or {}
        if bs_override:
            new_scheme.block_swap.enabled = bool(
                bs_override.get("enabled", new_scheme.block_swap.enabled)
            )

        logger.info(
            "acceleration_override 已应用: sage=%s, lora=%s, fbc=%s, blockswap=%s",
            new_scheme.acceleration.sage_attention.enabled,
            new_scheme.acceleration.lora.enabled if new_scheme.acceleration.lora else False,
            new_scheme.acceleration.first_block_cache.enabled if new_scheme.acceleration.first_block_cache else False,
            new_scheme.block_swap.enabled,
        )
        return new_scheme

    def _resolve_scheme_for_variant(self, variant_name: str) -> Optional[SchemeConfig]:
        """v0.6+：根据 variant 名反查适用方案。

        策略：
            1. SDNQ 4bit variant（名含 SDNQ / 4bit）：
               - 优先使用 state store 的默认方案（A/B/C/T8/T4）
               - 若默认方案不适用 SDNQ，则按显存自动选档（A/B/C，T 系不参与自动档）
            2. BF16 原版 variant（名不含 SDNQ / 4bit）：
               - 使用方案 D
            3. 都不匹配：按显存自动选档

        Args:
            variant_name: variant 名称

        Returns:
            SchemeConfig，失败返回 None
        """
        name_upper = variant_name.upper()
        is_sdnq = "SDNQ" in name_upper or "4BIT" in name_upper

        if is_sdnq:
            # SDNQ variant：优先用 state store 的默认方案
            # v0.7+：T8/T4 turbo 方案加入反查范围（用户显式选择的默认 turbo 方案优先生效）
            if self._default_scheme_id in ("A", "B", "C", "T8", "T4"):
                scheme = self.scheme_manager.load_scheme(self._default_scheme_id)
                if scheme and self.scheme_manager.validate_scheme_for_variant(scheme, variant_name):
                    logger.info(
                        "方案反查（SDNQ）: 使用默认方案 %s（来自 state store）",
                        scheme.scheme,
                    )
                    return scheme
            # 默认方案不适用，按显存自动选档
            scheme = self.scheme_manager.auto_select_scheme()
            if scheme and scheme.scheme in ("A", "B", "C"):
                logger.info(
                    "方案反查（SDNQ）: 自动选档 %s（默认方案不适用或未设置）",
                    scheme.scheme,
                )
                return scheme
            # 兜底：返回方案 A
            fallback = self.scheme_manager.load_scheme("A")
            logger.warning(
                "方案反查（SDNQ）: 自动选档失败，兜底使用方案 A",
            )
            return fallback
        else:
            # BF16 原版：使用方案 D
            scheme = self.scheme_manager.load_scheme("D")
            if scheme:
                logger.info("方案反查（BF16）: 使用方案 D")
                return scheme
            # D 不存在，按显存自动选档
            logger.warning("方案反查（BF16）: 方案 D 不存在，按显存自动选档")
            return self.scheme_manager.auto_select_scheme()

    async def _load_impl(self, variant: str) -> LoadResponse:
        """POST /load 的实际加载逻辑（已被 _load_in_background 替代）。

        保留此方法是为了兼容 BaseEngine 抽象接口。实际加载逻辑已移至
        _load_in_background。如果直接调用此方法（例如通过 super().load()），
        会执行同步加载（不推荐，会阻塞事件循环）。
        """
        variant_path = self.scanner.resolve_variant_path(variant)
        if variant_path is None:
            raise FileNotFoundError(
                f"variant 不存在: {variant} "
                f"（可用: {[v.name for v in self._available_variants]}）"
            )

        # v0.6+：解析方案
        scheme = self._resolve_scheme_for_variant(variant)
        if scheme is None:
            raise RuntimeError(f"无法为 variant={variant} 解析方案配置")

        logger.info(
            "开始加载 variant: %s (path=%s, scheme=%s)",
            variant, variant_path, scheme.scheme,
        )

        start_time = time.time()
        try:
            # v0.6+：传 scheme 而非 offload_level
            await asyncio.to_thread(
                self.wrapper.load,
                variant,                  # variant 名（存入 _loaded_variant）
                str(variant_path),        # variant 路径（兼容参数，实际用 scheme.model_subdir）
                scheme,                   # v0.6+ 方案配置
                self.config.device,
            )
        except Exception as e:
            logger.exception("MiniMax-H3 模型加载失败: variant=%s", variant)
            raise RuntimeError(f"MiniMax-H3 模型加载失败: {e}") from e

        load_time = time.time() - start_time
        vram_used = self.wrapper.get_vram_used()

        logger.info(
            "加载完成: variant=%s, scheme=%s, device=%s, offload=%s, load_time=%.2fs",
            variant, self.wrapper.loaded_scheme_id, self.wrapper.device,
            self.wrapper.offload_level, load_time,
        )

        return LoadResponse(
            variant=variant,
            state="ready",
            vram_used=f"{vram_used:.1f}MB",
            load_time=load_time,
        )

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload：卸载模型，释放显存。

        v0.6.1+ 修复：即使 wrapper 未加载模型（如 ERROR 状态下），
        也要确保状态机回到 IDLE，避免 ERROR 状态卡死无法恢复。
        """
        # 先处理状态机：若处于 ERROR 状态，调用 reset() 回到 IDLE
        if self.state_machine.state == RuntimeState.ERROR:
            try:
                self.state_machine.reset()
                logger.info("unload()：状态机从 ERROR 重置为 IDLE")
            except Exception as e:
                logger.warning("unload()：reset 状态机失败: %s", e)

        if not self.wrapper.is_loaded:
            # 确保状态机为 IDLE（可能从 LOADING/READY 来）
            if self.state_machine.state != RuntimeState.IDLE:
                try:
                    self.state_machine.transition(RuntimeState.IDLE)
                except Exception as e:
                    logger.warning("unload()：状态转换到 IDLE 失败: %s", e)
            return UnloadResponse(state="idle", vram_freed="0MB")

        vram_freed = self.wrapper.get_vram_used()
        await asyncio.to_thread(self.wrapper.unload)

        # 确保状态机为 IDLE
        if self.state_machine.state != RuntimeState.IDLE:
            try:
                self.state_machine.transition(RuntimeState.IDLE)
            except Exception as e:
                logger.warning("unload()：unload 后状态转换到 IDLE 失败: %s", e)

        logger.info("卸载完成: vram_freed=%.1fMB", vram_freed)
        return UnloadResponse(
            state="idle",
            vram_freed=f"{vram_freed:.1f}MB",
        )

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer：视频生成推理（SSE 流式推送进度）。

        流程：
            1. 生成 task_id（若 request.task_id 为 None）
            2. 解析 VideoInputs（从 request.inputs）
            3. 确定生成模式（task_type → mode）
            4. 提取图片/视频/音频输入（用于 I2V / Ref2VA / video_edit 模式）
            5. Context-IR 处理 prompt
            6. 上报 start / model_ready / prompt_processing 帧
            7. 线程池执行 wrapper.generate()（带心跳）
            8. 保存视频到 output/video/{task_id}.mp4
            9. 上报 completed 帧（含 video_url）

        错误处理：内部捕获异常，推送 error 帧后 return，
        不重新抛出（避免 BaseEngine.infer 把状态置为 ERROR）。
        """
        # ----- 0. 生成 task_id -----
        task_id = request.task_id or f"video_{uuid.uuid4().hex[:16]}"
        reporter = ProgressReporter(stream)

        # ----- 1. 解析输入 -----
        try:
            inputs = VideoInputs(**request.inputs)
        except Exception as e:
            logger.error("VideoInputs 解析失败: %s", e)
            reporter.emit_error(f"输入参数无效: {e}", "invalid_inputs")
            return

        # ----- 2. 上报任务创建到 gateway（v0.3.4+）-----
        reporter.set_task_id(task_id)
        await reporter.init_task_reporter()
        await reporter.report_created(
            request_summary={
                "prompt": inputs.prompt[:200] if inputs.prompt else "",
                "task_type": inputs.task_type,
                "duration": inputs.duration,
                "fps": inputs.fps,
                "size": f"{inputs.width}x{inputs.height}",
            },
            variant=request.variant or "",
            task_type="video_generation",  # v0.2.16+ 任务中心筛选语义
        )

        # ----- 3. 上报开始阶段 -----
        reporter.emit_start(task_id, inputs.prompt)
        reporter.emit_model_ready(
            self.wrapper.loaded_variant or "unknown",
            self.wrapper.device,
        )

        # ----- 4. 模式参数校验 -----
        mode = inputs.task_type
        if mode not in ("t2v", "i2v", "fl2v", "ref2v", "ref2va", "video_edit"):
            reporter.emit_error(
                f"未知生成模式: {mode}（支持: t2v / i2v / fl2v / ref2v / ref2va / video_edit）",
                "invalid_mode",
            )
            return

        # I2V / FL2V / Ref2VA 模式需要图片输入
        if mode in ("i2v", "fl2v", "ref2v") and not request.images:
            reporter.emit_error(
                f"{mode} 模式需要图片输入（通过 images 字段传入）",
                "missing_image",
            )
            return

        # video_edit 模式需要视频输入
        if mode == "video_edit" and not request.videos:
            reporter.emit_error(
                "video_edit 模式需要视频输入（通过 videos 字段传入）",
                "missing_video",
            )
            return

        # ref2va 多模态参考模式：独立校验链（R1/R2/R3）
        if mode == "ref2va":
            # R1：音频参考不能单独使用，必须配对至少 1 张图片或 1 个视频
            if not request.images and not request.videos:
                if request.audios:
                    reporter.emit_error(
                        "音频参考不能单独使用，请至少配对 1 张图片或 1 个视频参考",
                        "audio_without_visual",
                    )
                else:
                    reporter.emit_error(
                        "ref2va 模式至少需要 1 个图片或视频参考",
                        "missing_reference",
                    )
                return
            # R2：数量上限 图≤9 / 视频≤3 / 音频≤3
            if len(request.images) > 9 or len(request.videos) > 3 or len(request.audios) > 3:
                reporter.emit_error(
                    f"ref2va 参考数量超限（当前 图 {len(request.images)} / "
                    f"视频 {len(request.videos)} / 音频 {len(request.audios)}，"
                    f"上限 图≤9 / 视频≤3 / 音频≤3）",
                    "reference_limit",
                )
                return
            # R3：多模态参考必须走 Ref2VA variant（transformer_ref 分区）
            if self.wrapper.loaded_variant:
                try:
                    paradigm = self.wrapper._detect_paradigm(self.wrapper.loaded_variant)
                except ValueError:
                    paradigm = None
                if paradigm != "Ref2VA":
                    reporter.emit_error(
                        f"ref2va 模式需要 Ref2VA variant（transformer_ref 分区），"
                        f"当前已加载: {self.wrapper.loaded_variant}（{paradigm or '未知范式'}）。"
                        f"请先卸载并加载 Ref2VA 变体（如 MiniMax/H3-Ref2VA-SDNQ）",
                        "wrong_variant",
                    )
                    return
            # 场景判定日志（按上传组合自动判定）
            _has_i, _has_v, _has_a = bool(request.images), bool(request.videos), bool(request.audios)
            if _has_i and _has_v and _has_a:
                _scene = "全要素参考（身份 + 动作 + 音色）"
            elif _has_i and _has_v:
                _scene = "身份保持 + 动作迁移（图 + 视频）"
            elif _has_i and _has_a:
                _scene = "对口型 / 音色驱动（人物图 + 音频）"
            elif _has_v and _has_a:
                _scene = "动作参考 + 音色替换（视频 + 音频）"
            elif _has_i:
                _scene = "多图参考"
            else:
                _scene = "视频参考"
            logger.info(
                "ref2va 场景判定: %s（图 %d / 视频 %d / 音频 %d）",
                _scene, len(request.images), len(request.videos), len(request.audios),
            )

        # ----- 5. 提取多模态输入（用于 Context-IR 标签构造）-----
        image_refs = [
            img.url or img.path or ""
            for img in request.images
            if img.url or img.path
        ]
        video_refs = [
            vid.url or vid.path or ""
            for vid in request.videos
            if vid.url or vid.path
        ]
        audio_refs = [
            aud.url or aud.path or ""
            for aud in request.audios
            if aud.url or aud.path
        ]

        # ----- 6. Context-IR 处理 prompt -----
        reporter.emit_prompt_processing(inputs.prompt)
        enhanced_prompt = self.prompt_processor.process(
            prompt=inputs.prompt,
            task_type=mode,
            images=image_refs,
            videos=video_refs,
            audios=audio_refs,
        )

        # ----- 7. v0.6+ 解析当前已加载方案 -----
        # 推理参数优先级：inputs 显式传入 > scheme.inference 默认 > wrapper 默认
        # offload 级别由 scheme 决定（加载时固定），inputs.cpu_offload_level 不再生效
        current_scheme: Optional[SchemeConfig] = None
        if self.wrapper.loaded_scheme_id:
            current_scheme = self.scheme_manager.load_scheme(self.wrapper.loaded_scheme_id)
            if current_scheme:
                logger.info(
                    "推理使用已加载方案: %s（steps=%d, resolution=%s）",
                    current_scheme.scheme,
                    current_scheme.inference.num_inference_steps,
                    current_scheme.inference.default_resolution,
                )

        # ----- 8. v0.6+ 收集多模态输入 + 线程池执行推理（带心跳）-----
        # 通过 InferenceEngine 统一处理多模态输入收集和方案参数合并
        multimodal_inputs = self.inference_engine.collect_multimodal_inputs(
            mode=mode,
            images=request.images,
            videos=request.videos,
            audios=request.audios,
        )

        def _generate() -> dict:
            return self.inference_engine.generate(
                mode=mode,
                prompt=enhanced_prompt,
                scheme=current_scheme,
                negative_prompt=inputs.negative_prompt,
                duration=inputs.duration,
                fps=inputs.fps,
                width=inputs.width,
                height=inputs.height,
                num_inference_steps=inputs.num_inference_steps,
                guidance_scale=inputs.guidance_scale,
                seed=inputs.seed,
                # 45-专项：画质方案
                quality_scheme=inputs.quality_scheme,
                **multimodal_inputs,
            )

        # 心跳任务：generate() 是同步阻塞调用，期间定期推送心跳帧，
        # 重置 gateway 的 read timeout 计时。
        async def _heartbeat():
            elapsed = 0
            while True:
                await asyncio.sleep(30)
                elapsed += 30
                progress = min(0.80, 0.15 + elapsed * 0.005)
                reporter.emit_generating(progress, elapsed=elapsed)

        reporter.emit_generating(0.15, message="开始扩散去噪")
        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            result = await asyncio.to_thread(_generate)
        except Exception as e:
            logger.exception("MiniMax-H3 推理失败")
            reporter.emit_error(f"推理失败: {e}", "infer_failed")
            return
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # ----- 9. VAE 解码 + 音频解码（推理已包含，此处只上报进度）-----
        reporter.emit_vae_decoding()
        if mode in ("fl2v", "ref2v", "ref2va") and result.get("audio_data") is not None:
            reporter.emit_audio_decoding()

        # ----- 10. 保存视频 -----
        reporter.emit_saving()
        # 构造输出目录：{output_root}/video/（save_video 内部会自动创建）
        output_dir = self.config.output_root / "video"

        try:
            video_result = await asyncio.to_thread(
                save_video,
                result["video_frames"],
                result.get("audio_data"),
                output_dir,
                task_id,
                inputs.fps,
                sample_rate=result.get("sampling_rate", 32000),
            )
        except Exception as e:
            logger.exception("保存视频失败")
            reporter.emit_error(f"保存视频失败: {e}", "save_failed")
            return

        # ----- 10.5 推送合成警告（音视频时长不匹配等）-----
        mux_warnings = video_result.get("warnings", [])
        for w in mux_warnings:
            logger.warning("视频合成警告: %s", w)
            await reporter.emit_mux_warning(w)

        # ----- 10.8 45-专项：画质方案后处理（A 超分 / C 脸修复）-----
        # 采样增强（D）在生成阶段生效，后处理阶段只跑 A / C（顺序：先超分再脸修复）。
        metadata0 = result.get("metadata", {})
        qs_tokens: set[str] = set()
        _qs_raw = str(metadata0.get("quality_scheme") or "").strip().lower()
        if _qs_raw and _qs_raw != "none":
            qs_tokens = {t for t in _qs_raw.split("+") if t and t in {"a", "b", "c", "d"}}
        post_tokens = {t for t in qs_tokens if t in {"a", "c"}}
        post_info: Optional[dict] = None
        if post_tokens:
            saved_abs = str((output_dir / f"{task_id}.mp4").resolve())
            reporter.emit_generating(0.92, message="画质增强后处理（超分/脸修复）")
            post_info = await asyncio.to_thread(
                quality_postprocess.apply_quality_postprocess,
                post_tokens,
                saved_abs,
                output_dir,
                task_id,
            )
            for _st in post_info.get("stages", []):
                logger.info("[quality_scheme] 后处理阶段: %s", _st.get("detail"))

        # ----- 11. 上报完成 -----
        metadata = result.get("metadata", {})
        final_result = {
            "video_url": video_result["video_url"],
            "video_only_url": video_result["video_only_url"],
            "audio_url": video_result["audio_url"],
            "duration": video_result["duration"],
            "fps": video_result["fps"],
            "mode": mode,
            "variant": self.wrapper.loaded_variant,
            "scheme_id": self.wrapper.loaded_scheme_id,
            "device": self.wrapper.device,
            "offload_level": self.wrapper.offload_level,
            "seed": inputs.seed,
            "is_placeholder": metadata.get("is_placeholder", False),
            "warnings": mux_warnings,
            # 45-专项：透出"实际生效"参数到任务列表（请求值 ≠ 生效值，便于判断）
            "quality_scheme": metadata.get("quality_scheme", "B"),
            "effective_width": metadata.get("effective_width"),
            "effective_height": metadata.get("effective_height"),
            "num_inference_steps": metadata.get("num_inference_steps"),
            "guidance_scale": metadata.get("guidance_scale"),
        }

        # 45-专项：若后处理改变了最终产物，把对外 video_url 指向增强后文件，
        # 并保留原始 URL 供对比（receipts 溯源）。
        if post_info and post_info.get("final_path"):
            _final_post = str(Path(post_info["final_path"]).resolve())
            if Path(_final_post).is_file() and _final_post != saved_abs:
                final_result["video_original_url"] = final_result["video_url"]
                final_result["video_url"] = quality_postprocess._to_output_url(_final_post)
                final_result["quality_scheme_post"] = post_info.get("stages", [])
                logger.info(
                    "画质方案后处理完成，最终产物: %s (原始 %s)",
                    final_result["video_url"], final_result["video_original_url"],
                )

        logger.info(
            "视频生成完成: task_id=%s, mode=%s, duration=%.2fs, video_url=%s",
            task_id, mode, video_result["duration"], video_result["video_url"],
        )

        await reporter.emit_completed(task_id, final_result)

    async def _status_impl(self) -> StatusResponse:
        """GET /status：返回引擎状态。"""
        vram_used = self.wrapper.get_vram_used()
        loaded_loras = [item["name"] for item in self.wrapper.lora_manager.get_loaded_loras()]
        # v0.4.x：device_name 给 UI 显示友好标签（GPU (CUDA) / CPU），
        # device 仍保留原始字符串（cuda:0 / cpu）供程序使用
        raw_device = self.wrapper.device
        if raw_device.startswith("cuda"):
            device_name = "GPU (CUDA)"
        elif raw_device == "cpu":
            device_name = "CPU"
        else:
            device_name = raw_device
        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self.wrapper.loaded_variant,
            vram_used=f"{vram_used:.1f}MB",
            loaded_loras=loaded_loras,
            last_used_at=self.state_machine.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "device": raw_device,
                "device_name": device_name,
                "offload_level": self.wrapper.offload_level,
                "is_loaded": self.wrapper.is_loaded,
                "available_loras": self._available_loras,
                # v0.6+ 方案信息
                "loaded_scheme_id": self.wrapper.loaded_scheme_id,
                "default_scheme_id": self._default_scheme_id,
                "available_schemes": [
                    {
                        "id": s.scheme,
                        "name": s.name,
                        "description": s.description,
                        "vram_required": s.vram_required,
                        "precision": s.precision,
                    }
                    for s in self.scheme_manager.list_schemes()
                ],
            },
        )

    # ============ v0.6+ 方案管理 API ============

    def list_schemes(self) -> list[dict]:
        """v0.6+：列出所有可用方案（供 UI / 平台 API 调用）。

        Returns:
            方案信息列表（每项含 id / name / description / vram_required / precision）
        """
        return [
            {
                "id": s.scheme,
                "name": s.name,
                "description": s.description,
                "vram_required": s.vram_required,
                "precision": s.precision,
                "size_gb": s.size_gb,
                "offload_level": s.offload_level,
                "model_repo": s.model_repo,
                "applicable_variants": s.applicable_variants,
            }
            for s in self.scheme_manager.list_schemes()
        ]

    def get_default_scheme(self) -> Optional[str]:
        """v0.6+：获取当前默认方案 ID。"""
        return self._default_scheme_id

    def set_default_scheme(self, scheme_id: str) -> bool:
        """v0.6+：设置默认方案并持久化。

        Args:
            scheme_id: 方案 ID（A / B / C / D）

        Returns:
            是否设置成功
        """
        scheme = self.scheme_manager.load_scheme(scheme_id)
        if scheme is None:
            logger.warning("设置默认方案失败: 方案 %s 不存在", scheme_id)
            return False
        self._default_scheme_id = scheme_id
        self.scheme_store.set_default_scheme(scheme_id)
        logger.info("默认方案已设置并持久化: %s（%s）", scheme_id, scheme.name)
        return True

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load：加载 LoRA 到 transformer 模型。"""
        try:
            # 从 LoRA 目录解析路径
            lora_path = self.config.model_dir / "loras" / f"{name}.safetensors"
            if not lora_path.exists():
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=404,
                    detail=f"LoRA 文件不存在: {lora_path}（可用: {self._available_loras}）",
                )

            await asyncio.to_thread(self.wrapper.load_lora, str(lora_path), weight)
            loaded_loras = [item["name"] for item in self.wrapper.lora_manager.get_loaded_loras()]
            return LoraLoadResponse(
                name=name,
                weight=weight,
                loaded_loras=loaded_loras,
            )
        except Exception as e:
            logger.exception("LoRA 加载失败: name=%s", name)
            raise RuntimeError(f"LoRA 加载失败: {e}") from e

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload：卸载 LoRA。"""
        try:
            await asyncio.to_thread(self.wrapper.unload_lora, name)
            loaded_loras = [item["name"] for item in self.wrapper.lora_manager.get_loaded_loras()]
            return LoraUnloadResponse(
                name=name,
                loaded_loras=loaded_loras,
            )
        except Exception as e:
            logger.exception("LoRA 卸载失败: name=%s", name)
            raise RuntimeError(f"LoRA 卸载失败: {e}") from e
