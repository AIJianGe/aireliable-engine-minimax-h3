"""MiniMax-H3 模型加载与推理核心封装（多方案版 v0.6+）。

封装 MiniMax-H3 的模型加载、卸载、推理流程，提供线程安全接口。

v0.6+ 多方案架构升级（v2.0）：
    - 按 SchemeConfig 加载模型（SDNQ 4bit / BF16 双套）
    - 加载完成后自动注入加速方案（SageAttention / LightX2V LoRA / FirstBlockCache / BlockSwap）
    - 卸载时自动恢复所有 monkey-patch
    - 支持方案运行时切换（需先 unload 再 load 新方案）

设计要点：
    - 线程安全：所有加载/推理/卸载操作用 _lock 保护
    - CPU offload 分级：L0/L1/L2/L3 四级显存优化策略
    - variant 范式识别：FL2VA（文本直接生成含音频视频）/ Ref2VA（参考图生成含音频视频）
    - 文件完整性校验：load() 只校验不下载，缺失文件抛 RuntimeError 指引用户下载
    - 真实推理：基于 diffusers 0.40.0.dev0 的 ModularPipeline
    - 加速方案注入：通过 accelerators.patcher 统一管理（遵循 project-rules §3 try/finally）

模型组件（基于 MiniMaxAI/MiniMax-H3 diffusers 格式）：
    - transformer/transformer_ref:   视频扩散 transformer（FL2VA 用 transformer/，Ref2VA 用 transformer_ref/）
    - text_encoder:                   Qwen3VL-32B 文本编码器（共享）
    - vae:                            视频 VAE（共享，潜空间 ↔ 像素帧）
    - audio_vae:                       音频 VAE（共享，潜空间 ↔ 音频波形）
    - scheduler/audio_scheduler:      调度器（共享）
    - processor/tokenizer:            处理器与分词器（共享）

权重格式（v0.6+ 双套）：
    - SDNQ 4bit（方案 A/B/C）：models/video/minimax-h3/sdnq_format/
      仓库：OzzyGT/MiniMax_H3_sdnq_dynamic_4bit
    - BF16 原版（方案 D）：models/video/minimax-h3/diffusers_format/
      仓库：MiniMaxAI/MiniMax-H3
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .lora_manager import LoRAManager
from .scheme_manager import SchemeConfig

logger = logging.getLogger(__name__)


# ===== 各 offload 级别所需显存阈值（GB） =====
# 与 config.py 的 _OFFLOAD_VRAM_THRESHOLDS 保持一致
_OFFLOAD_VRAM_THRESHOLDS = {
    "L0": 48.0,   # 不卸载，全 GPU
    "L1": 35.0,   # 轻度卸载（TE 编码后移到 CPU）
    "L2": 24.0,   # 中度卸载（accelerate 层卸载）
}


# ===== variant 范式识别 =====
# FL2VA: Full Latent to Video Audio（文本直接生成含音频的视频）
# Ref2VA: Reference to Video Audio（参考图 + 文本生成含音频的视频）
_FL2VA_VARIANTS = ("FL2VA",)
_REF2VA_VARIANTS = ("Ref2VA",)

# v1.2+（41 号任务）latent NaN/Inf 发散判定阈值（占比）。
# 超过该占比视为"扩散数值发散"（latents 已成垃圾），直接抛错落 failed，绝不替换为 0 输出灰屏；
# 低于该占比视为偶发噪声点，nan_to_num 兜底避免全黑视频。
_NAN_FAIL_FRACTION = 0.001  # 0.1%


# diffusers 格式权重子目录名
_DIFFUSERS_FORMAT_SUBDIR = "diffusers_format"


class MiniMaxH3Wrapper:
    """MiniMax-H3 模型封装器（真实推理版）。

    封装 MiniMax-H3 的加载/推理/卸载流程，提供：
        - 线程安全的模型加载/卸载
        - CPU offload 分级（L0/L1/L2/L3）
        - FL2VA / Ref2VA 范式自动识别
        - 多 LoRA 运行时管理（接口保留，实现待完善）
        - 显存预检与监控
        - 真实推理调用 diffusers ModularPipeline
    """

    def __init__(
        self,
        models_root: Path,
        model_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化封装器。

        Args:
            models_root: 模型根目录（如 models/）
            model_dir: MiniMax-H3 模型目录（如 models/video/minimax-h3）
            logger: 可选的日志器
        """
        self.models_root = models_root
        self.model_dir = model_dir
        self._logger = logger or logging.getLogger(__name__)

        # 线程锁：保护加载/推理/卸载互斥
        self._lock = threading.Lock()

        # ModularPipeline 实例（load 后填充）
        self._pipe: Any = None
        self._loaded_variant: Optional[str] = None
        # v0.4.x：初始化时即探测可用设备，避免 idle 状态下 /status 显示 device=cpu 误导用户
        # load() 调用时会根据传入的 device 参数再次解析并覆盖
        self._device: str = self._resolve_device("auto")
        self._offload_level: str = "L1"
        # v0.6+：当前已加载方案（SchemeConfig.scheme，如 "A" / "D"），用于 /status 显示
        self._loaded_scheme_id: Optional[str] = None

        # LoRA 管理器（运行时 LoRA 加载/卸载）
        lora_dir = model_dir / "loras"
        self._lora_manager = LoRAManager(lora_dir, logger=self._logger)
        # v0.7+：用户手动挂载的 LoRA 旁路上下文（name → LoRAContext，load_lora/unload_lora 用）
        self._manual_lora_ctxs: dict[str, Any] = {}

        # v0.6+ 加速方案上下文（用于卸载时恢复所有 monkey-patch）
        # 延迟导入避免循环依赖；类型用 Any 避免 accelerators 模块未就绪时 import 失败
        self._acceleration_ctx: Any = None

        # v0.6.5+ 缓存 SDNQ 量化的组件（同一 variant 重载时复用，避免重复加载）
        # v0.6.5+ 修订根因：此前认为 sdnq 库第二次调用崩溃，实际是 RAM 耗尽：
        #   text_encoder(21GB) + transformer(18.8GB) + transformer_ref(18.8GB) = 58.6GB
        #   系统总 RAM 64GB，预加载三个模型会 OOM 崩溃。
        # v0.6.5+ 策略：
        #   1. 移除预加载机制（避免同时加载三个模型）
        #   2. 卸载时清空缓存（释放 RAM，允许切换 variant）
        #   3. 同一 variant 重载时复用缓存（跳过 sdnq.load_sdnq_model）
        self._cached_sdnq_components: dict[str, Any] = {}

        # v0.7.7+ T2.2 conditioning 磁盘缓存（load 时创建，跨重启复用；未加载时为 None）
        self._cond_cache: Any = None
        # 当前请求的 conditioning 缓存 key（由 generate() 计算，TE 编码 patch 消费；
        # 仅单次生成有效，推理完成后置 None 防止残留误用）
        self._cond_cache_key: Optional[str] = None

        self._logger.info(
            "MiniMaxH3Wrapper 初始化: model_dir=%s, lora_dir=%s, device=%s",
            model_dir, lora_dir, self._device,
        )

    # ============ 属性 ============

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。"""
        return self._pipe is not None

    @property
    def loaded_variant(self) -> Optional[str]:
        """已加载的变体名。"""
        return self._loaded_variant

    @property
    def device(self) -> str:
        """当前推理设备。"""
        return self._device

    @property
    def offload_level(self) -> str:
        """当前 CPU offload 级别。"""
        return self._offload_level

    @property
    def loaded_scheme_id(self) -> Optional[str]:
        """v0.6+：当前已加载方案 ID（如 "A" / "D"），未加载返回 None。"""
        return self._loaded_scheme_id

    @property
    def lora_manager(self) -> LoRAManager:
        """LoRA 管理器。"""
        return self._lora_manager

    # ============ 加载 ============

    def load(
        self,
        variant_name: str,
        variant_path: str,
        scheme: SchemeConfig,
        device: str = "auto",
    ) -> None:
        """加载 MiniMax-H3 模型（基于 diffusers ModularPipeline + 方案配置）。

        v0.6+ 多方案版：按 SchemeConfig 决定加载的模型目录、精度、offload 级别、加速方案。

        Args:
            variant_name: variant 名称（如 MiniMax/H3-FL2VA-SDNQ / MiniMax/H3-FL2VA）
            variant_path: 保留参数（兼容旧接口，实际使用 scheme.model_subdir）
            scheme: 方案配置（SchemeConfig 实例，决定 model_subdir / precision / offload / 加速方案）
            device: 推理设备（auto / cpu / cuda:0）

        Raises:
            RuntimeError: 加载失败 / 文件缺失 / OOM

        加载流程：
            1. 解析 device（auto → cuda:0 / cpu）
            2. 解析 variant 范式（FL2VA → transformer/ / Ref2VA → transformer_ref/）
            3. 解析模型目录（scheme.model_subdir 相对 model_dir）
            4. 校验模型目录与关键文件（不下载，缺失抛 RuntimeError）
            5. 显存预检（按 scheme.vram_required 阈值）
            6. ModularPipeline.from_pretrained()（不传 workflow，保留所有条件分支）
            7. pipe.load_components(names=[...], dtype=按 scheme.precision 决定)
            8. 根据 scheme.offload_level 应用加载策略
            9. v0.6+ 注入加速方案（SageAttention / LoRA / FirstBlockCache / BlockSwap）
        """
        with self._lock:
            if self._pipe is not None:
                self._logger.warning(
                    "模型已加载（variant=%s, scheme=%s），先卸载",
                    self._loaded_variant, self._loaded_scheme_id,
                )
                self._unload_unsafe()

            # ----- 1. 解析 device & offload（从 scheme 取，忽略旧 offload_level 参数） -----
            self._device = self._resolve_device(device)
            self._offload_level = scheme.offload_level
            self._loaded_scheme_id = scheme.scheme
            self._logger.info(
                "开始加载 MiniMax-H3 模型: variant=%s, scheme=%s, precision=%s, device=%s, offload=%s",
                variant_name, scheme.scheme, scheme.precision,
                self._device, self._offload_level,
            )

            # ----- 2. 解析 variant 范式 -----
            # paradigm 决定加载哪个 transformer 分区：
            #   - FL2VA:  加载 transformer/   → 支持 T2V/I2V/FL2V
            #   - Ref2VA: 加载 transformer_ref/ → 支持 Ref2V/VideoEdit
            # 不再使用 diffusers 的 workflow 机制（会触发 _workflow_map 缺失错误）
            # 改为通过 load_components(names=[...]) 显式指定要加载的组件
            paradigm = self._detect_paradigm(variant_name)
            self._logger.info("variant 范式: %s", paradigm)

            # ----- 3. 解析模型目录（scheme.model_subdir 相对 model_dir） -----
            # v0.6+ 双套目录：
            #   - SDNQ 4bit（方案 A/B/C）：model_dir / "sdnq_format"
            #   - BF16 原版（方案 D）：model_dir / "diffusers_format"
            diffusers_dir = self.model_dir / scheme.model_subdir
            if not diffusers_dir.exists():
                raise RuntimeError(
                    f"模型目录不存在: {diffusers_dir}\n"
                    f"方案 {scheme.scheme}（precision={scheme.precision}）需要下载模型到该目录。\n"
                    f"HuggingFace 仓库：{scheme.model_repo}\n"
                    f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，选择镜像源后下载。\n"
                    f"或手动下载命令：\n"
                    f"  hf download {scheme.model_repo} --local-dir {diffusers_dir} \\\n"
                    f"    --include \"transformer/*\" --include \"transformer_ref/*\" \\\n"
                    f"    --include \"text_encoder/*\" --include \"vae/*\" --include \"audio_vae/*\" \\\n"
                    f"    --include \"scheduler/*\" --include \"audio_scheduler/*\" \\\n"
                    f"    --include \"processor/*\" --include \"tokenizer/*\" \\\n"
                    f"    --include \"model_index.json\" --include \"modular_model_index.json\""
                )

            self._check_files_integrity(variant_name, diffusers_dir, paradigm, scheme)

            # ----- 4. 显存预检（按方案要求） -----
            required_gb = self._parse_vram_required(scheme.vram_required)
            # 最低 8GB 才能加载配置文件；实际显存需求由 scheme 决定
            self._check_vram_available(self._device, required_gb=max(8.0, required_gb * 0.3))

            # ----- 4.5 T2.2 conditioning 磁盘缓存（随 variant 目录，跨重启复用） -----
            try:
                from .conditioning_cache import ConditioningCache
                self._cond_cache = ConditioningCache(
                    self.model_dir / "conditioning_cache",
                    logger=self._logger,
                )
            except Exception as _ce:
                self._logger.warning("[cond-cache] 初始化失败（缓存禁用）: %s", _ce)
                self._cond_cache = None

            # ----- 5. 加载 pipeline -----
            start_time = time.time()
            try:
                from diffusers import ModularPipeline
                import torch

                # 强制离线模式：跳过 huggingface.co 文件更新检查
                # （权重已下载到本地，每次启动都 HEAD 检查会因网络问题超时）
                import os
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

                # 创建 pipeline（仅加载配置，不加载权重）
                # 关键：不传 workflow 参数！
                # - 传 workflow 会让 from_pretrained 内部调用 blocks.get_workflow(workflow)
                #   将 blocks 裁剪为基类 SequentialPipelineBlocks（无 _workflow_map）
                # - 之后 load_components(workflow=...) 会再次调用 blocks.get_workflow()
                #   报错 "_workflow_map is not set for SequentialPipelineBlocks"
                # - 不传 workflow 保留所有条件分支（t2va/fl2va/ref2va），运行时由
                #   ConditionalPipelineBlocks.select_block 根据实际输入选择执行分支
                self._logger.info(
                    "创建 ModularPipeline: path=%s (不传 workflow，保留所有条件分支)",
                    diffusers_dir,
                )
                t0 = time.time()
                self._pipe = ModularPipeline.from_pretrained(
                    str(diffusers_dir),
                )
                # v0.7.9+ 修复：Ref2V/Ref2VA 长 token 编码卡死根因。
                # 官方释放版把图片参考无条件按「短边 2048、放大且无面积上限」编码
                # （before_encoder.py 归一化：scale = reference_image_short_edge / min(w,h)），
                # 3 张参考图即产生 ~21621 个视觉 token；T2.1 显存预检（_needed_mb =
                # 4096 + token*2 ≈ 47GB）判 GPU 放不下 → 本次降级 CPU 编码，
                # CPU 编码 21621 token 实测 30min+ 无扩散 step 卡死。
                # 修法：覆写参考图短边，把单张 token 压到 GPU + SDPA 可秒级编码的范围，
                # 同时保留参考图主体细节（768 与生成画布短边一致）。
                # 注意：fl2va/i2v 走 image 首帧、不受 references 影响，此处不动它们。
                self._pipe.config.reference_image_short_edge = 768
                self._logger.warning(
                    "已覆写 reference_image_short_edge=768（原 %s，压低 Ref2V/Ref2VA 参考图 token 数）",
                    2048,
                )
                self._logger.info(
                    "ModularPipeline 创建完成: 耗时=%.2fs", time.time() - t0,
                )

                # 加载组件（权重）
                # 关键1：必须传入 pretrained_model_name_or_path 覆盖 modular_model_index.json
                # 中的 HuggingFace 仓库 ID（"MiniMaxAI/MiniMax-H3"），否则 load_components
                # 会尝试从网络加载（HF_HUB_OFFLINE=True 时全部失败，组件为 None）
                # 传入本地 diffusers_dir 后，每个组件的 subfolder（如 "text_encoder"）
                # 会拼接到本地路径下，从本地加载权重
                # 关键2：不传 workflow，改用 names=[...] 显式指定要加载的组件列表
                # 这样只加载当前范式所需的 transformer 分区（transformer/ 或 transformer_ref/）
                # 避免同时加载两个分区（共 123GB）导致 OOM
                # 关键3：dtype 按 scheme.precision 决定（BF16 原版用 torch.bfloat16，
                # SDNQ 4bit 量化版也用 torch.bfloat16 加载，量化由权重文件本身决定）
                # 确定当前范式对应的 transformer 组件名与子目录
                # FL2VA → transformer/ 分区；Ref2VA → transformer_ref/ 分区
                if paradigm == "FL2VA":
                    transformer_attr = "transformer"
                    transformer_subdir = "transformer"
                else:  # Ref2VA
                    transformer_attr = "transformer_ref"
                    transformer_subdir = "transformer_ref"

                # v0.6.5+ 优化加载顺序：先加载大模型（SDNQ text_encoder 19.6GB + transformer 17.5GB），
                # 再加载共享组件（vae 5.2GB + audio_vae 0.6GB）。
                # 原顺序（共享→text_encoder→transformer）峰值 RAM ~46GB，在 48GB 可用 RAM 上 OOM。
                # 新顺序（text_encoder→transformer→共享）峰值 RAM ~43GB，留 5GB 余量。
                # BF16 模式不受影响（所有组件通过 diffusers load_components 一次性加载）。
                # v0.7+ 精度解析改为 _resolve_activation_dtype：支持 sdnq_uint4_fp16
                # 显式 fp16 档（T 系 turbo 方案）+ Turing 等无 BF16 Tensor Core
                # 的 GPU 自动降级 fp16
                load_dtype = self._resolve_activation_dtype(scheme.precision)

                if scheme.precision in ("sdnq_uint4", "sdnq_uint4_fp16"):
                    # ===== SDNQ 模式：先加载大模型，再加载共享组件 =====

                    # v0.7.2+ 修复：SDNQ uint4 全黑视频溢出问题
                    # 根因：torch._int_mm 输出 INT32（范围 ±~1e7），
                    # 直接转 fp16（最大 ±65504）溢出 → Inf → 乘 scale 后 NaN →
                    # latent NaN → VAE 解码全 0 → 全黑视频。
                    # 修复：monkey-patch int_scaled_mm_torch，INT32 → fp32 做中间计算，再转 out_dtype。
                    # 保持 SDNQ 层参数 dtype 不变，避免 dtype 不匹配。
                    try:
                        import sdnq.kernel_wrappers as kernel_wrappers
                        import torch

                        # 保存原始函数（用于卸载时恢复）
                        if not hasattr(self, '_orig_int_scaled_mm_torch'):
                            self._orig_int_scaled_mm_torch = kernel_wrappers.int_scaled_mm_torch
                            self._logger.debug("已保存原始 int_scaled_mm_torch 用于后续恢复")

                        def patched_int_scaled_mm_torch(a, b, scale_a, scale_b, bias=None, out_dtype=torch.float32):
                            # 使用 fp32 中间精度避免 INT32→fp16 溢出：
                            # - Triton/Triton 路径返回 fp32（安全，无溢出）
                            # - PyTorch Eager 回退路径：torch._int_mm 返回 INT32，
                            #   .to(fp32) 不溢出（fp32 可精确表示到 ~16.7M，INT32 实际范围 ±~1e7）
                            # - 最后转目标 dtype（fp16/bf16）
                            result = kernel_wrappers.int_mm_func(a, b, out_dtype=torch.float32)
                            if bias is None:
                                return result.mul_(scale_a).mul_(scale_b).to(dtype=out_dtype)
                            else:
                                return torch.addcmul(bias, result.mul_(scale_a), scale_b).to(dtype=out_dtype)

                        kernel_wrappers.int_scaled_mm_torch = patched_int_scaled_mm_torch
                        self._logger.info("已 monkey-patch sdnq.kernel_wrappers.int_scaled_mm_torch 解决 INT32→fp16 溢出问题")

                        # v0.8+ 修复：fp_mm_torch 中 fp16 矩阵乘法溢出
                        # 根因：fp_mm_torch 使用 fp16 的 torch.mm 做矩阵乘法，
                        # 当 dim=4096 时中间结果可达 65536，超出 fp16 最大值 65504 → Inf。
                        # 修复：改用 fp32 矩阵乘法，避免中间精度损失。
                        # 调用链：fp16_matmul → fp_scaled_mm_func → fp_scaled_mm_torch → fp_mm_func → fp_mm_torch
                        if not hasattr(self, '_orig_fp_mm_torch'):
                            self._orig_fp_mm_torch = kernel_wrappers.fp_mm_torch
                            self._logger.debug("已保存原始 fp_mm_torch 用于后续恢复")

                        def patched_fp_mm_torch(a, b, out_dtype=torch.float32):
                            # 使用 fp32 矩阵乘法避免 fp16 溢出
                            a = a.to(dtype=torch.float32)
                            b = b.to(dtype=torch.float32)
                            return torch.mm(a, b).to(dtype=out_dtype)

                        kernel_wrappers.fp_mm_torch = patched_fp_mm_torch
                        self._logger.info("已 monkey-patch sdnq.kernel_wrappers.fp_mm_torch 解决 fp16 矩阵乘法溢出问题")
                    except Exception as _patch_e:
                        self._logger.warning("monkey-patch sdnq.kernel_wrappers 失败: %s", _patch_e)

                    # 1. SDNQ text_encoder（19.6GB）
                    t_te = time.time()
                    self._logger.info(
                        "SDNQ 加载分支: 通过 sdnq.load_sdnq_model 加载 text_encoder/ (precision=%s, dtype=%s)",
                        scheme.precision, load_dtype,
                    )
                    self._load_sdnq_component(
                        diffusers_dir=diffusers_dir,
                        subdir="text_encoder",
                        attr="text_encoder",
                        dtype=load_dtype,
                        model_cls=None,  # 自动推断（规避 sdnq UnboundLocalError）
                    )
                    self._logger.info(
                        "text_encoder 加载完成: 耗时=%.2fs", time.time() - t_te,
                    )

                    # 2. SDNQ transformer（17.5GB）
                    t_tr = time.time()
                    self._logger.info(
                        "SDNQ 加载分支: 通过 sdnq.load_sdnq_model 加载 %s/ (precision=%s, dtype=%s)",
                        transformer_subdir, scheme.precision, load_dtype,
                    )
                    self._load_sdnq_component(
                        diffusers_dir=diffusers_dir,
                        subdir=transformer_subdir,
                        attr=transformer_attr,
                        dtype=load_dtype,
                        model_cls=None,  # _load_sdnq_component 内部自动推断
                    )
                    self._logger.info(
                        "transformer 加载完成: 耗时=%.2fs", time.time() - t_tr,
                    )

                    # v1.2 诊断探针：检查 SDNQ 去量化后权重是否含 NaN/Inf（定位全 NaN 输出根因）
                    try:
                        _transformer = getattr(self._pipe, transformer_attr)
                        _tr_bad = {"count": 0, "total": 0, "first": None}
                        for _name, _mod in _transformer.named_modules():
                            _dq = getattr(_mod, "sdnq_dequantizer", None)
                            if _dq is None:
                                continue
                            _tr_bad["total"] += 1
                            try:
                                _w = _dq(
                                    _mod.weight, _mod.scale,
                                    zero_point=getattr(_mod, "zero_point", None),
                                    svd_up=getattr(_mod, "svd_up", None),
                                    svd_down=getattr(_mod, "svd_down", None),
                                )
                                _nan = float(_w.isnan().float().mean())
                                _inf = float(_w.isinf().float().mean())
                                if _nan > 0 or _inf > 0:
                                    _tr_bad["count"] += 1
                                    if _tr_bad["first"] is None:
                                        _tr_bad["first"] = _name
                                    self._logger.warning(
                                        "[权重探针] sdnq层 %s: nan=%.4f inf=%.4f dtype=%s shape=%s scale_nan=%.4f",
                                        _name, _nan, _inf, _w.dtype, list(_w.shape),
                                        float(_mod.scale.isnan().float().mean()) if _mod.scale is not None else -1,
                                    )
                            except Exception as _wee:
                                self._logger.warning("[权重探针] 层 %s 去量化异常（忽略）: %s", _name, _wee)
                        self._logger.warning(
                            "[权重探针] transformer sdnq 线性层总计 %d，含 NaN/Inf 权重 %d 个，首个=%s",
                            _tr_bad["total"], _tr_bad["count"], _tr_bad["first"],
                        )
                    except Exception as _wprobe_e:
                        self._logger.warning("[权重探针] 整体探针异常: %s", _wprobe_e)

                    # 3. 共享组件（vae 5.2GB + audio_vae 0.6GB + tokenizer/processor/scheduler ~0GB）
                    component_names = [
                        "tokenizer", "processor",
                        "vae", "audio_vae",
                        "scheduler", "audio_scheduler",
                    ]
                    t0 = time.time()
                    self._logger.info(
                        "开始加载共享组件: dtype=%s (precision=%s), paradigm=%s, names=%s, local_path=%s",
                        load_dtype, scheme.precision, paradigm, component_names, diffusers_dir,
                    )
                    try:
                        self._pipe.load_components(
                            names=component_names,
                            dtype=load_dtype,
                            pretrained_model_name_or_path=str(diffusers_dir),
                        )
                    except Exception as _load_e:
                        self._logger.exception("load_components 调用失败（共享组件）")
                        raise RuntimeError(f"共享组件加载失败: {_load_e}") from _load_e
                    self._logger.info(
                        "共享组件加载完成: 耗时=%.2fs", time.time() - t0,
                    )

                else:
                    # ===== BF16 模式：所有组件通过 diffusers load_components 一次性加载 =====
                    component_names = [
                        "text_encoder", "tokenizer", "processor",
                        "vae", "audio_vae",
                        "scheduler", "audio_scheduler",
                    ]
                    t0 = time.time()
                    self._logger.info(
                        "开始加载共享组件(含text_encoder): dtype=%s (precision=%s), paradigm=%s, names=%s, local_path=%s",
                        load_dtype, scheme.precision, paradigm, component_names, diffusers_dir,
                    )
                    try:
                        self._pipe.load_components(
                            names=component_names,
                            dtype=load_dtype,
                            pretrained_model_name_or_path=str(diffusers_dir),
                        )
                    except Exception as _load_e:
                        self._logger.exception("load_components 调用失败（共享组件）")
                        raise RuntimeError(f"共享组件加载失败: {_load_e}") from _load_e
                    self._logger.info(
                        "共享组件加载完成: 耗时=%.2fs", time.time() - t0,
                    )

                    # BF16 transformer 通过 diffusers load_components 加载
                    t_tr = time.time()
                    self._logger.info(
                        "BF16 加载分支: 通过 load_components 加载 %s/ (precision=%s)",
                        transformer_subdir, scheme.precision,
                    )
                    try:
                        self._pipe.load_components(
                            names=[transformer_attr],
                            dtype=load_dtype,
                            pretrained_model_name_or_path=str(diffusers_dir),
                        )
                    except Exception as _load_e:
                        self._logger.exception("load_components 调用失败（transformer）")
                        raise RuntimeError(f"transformer 加载失败: {_load_e}") from _load_e
                    self._logger.info(
                        "transformer 加载完成: 耗时=%.2fs", time.time() - t_tr,
                    )

                # v0.6.5+ 移除预加载机制：
                # 此前预加载另一个 transformer 分区以"避免 sdnq 库第二次调用崩溃"，
                # 但实际根因是 RAM 耗尽（3 个模型共 58.6GB > 系统 64GB 可用 RAM）。
                # 移除后：
                #   - 单个 variant 加载仅需 text_encoder(21GB) + transformer(18.8GB) = 39.8GB ✓
                #   - 卸载时清空缓存释放 RAM，切换 variant 时重新加载（参见 _unload_unsafe）
                #   - 同一 variant 重载时复用缓存（跳过 sdnq.load_sdnq_model）

                # 校验关键组件加载成功（防止静默失败导致推理时报 NoneType 错误）
                _all_components = (
                    ["text_encoder", "tokenizer", "processor", "vae", "scheduler"]
                    + (["transformer"] if paradigm == "FL2VA" else ["transformer_ref"])
                    + (["audio_vae", "audio_scheduler"])
                )
                _missing_components = []
                for _comp_name in _all_components:
                    if getattr(self._pipe, _comp_name, None) is None:
                        _missing_components.append(_comp_name)
                if _missing_components:
                    raise RuntimeError(
                        f"组件加载失败（值为 None）: {_missing_components}\n"
                        f"可能原因：modular_model_index.json 中的 pretrained_model_name_or_path "
                        f"指向 HuggingFace 仓库 ID 而非本地路径，且 HF_HUB_OFFLINE=1。\n"
                        f"本地路径：{diffusers_dir}"
                    )
                # 打印每个组件的类型，便于调试
                for _comp_name in _all_components:
                    _comp_val = getattr(self._pipe, _comp_name, None)
                    if _comp_val is not None:
                        self._logger.info(
                            "  组件 %s: type=%s",
                            _comp_name, type(_comp_val).__name__,
                        )

                # ----- 6. 根据 scheme.offload_level 应用加载策略 -----
                t0 = time.time()
                self._apply_offload_strategy(self._offload_level)
                self._logger.info(
                    "offload 策略应用完成: level=%s, 耗时=%.2fs",
                    self._offload_level, time.time() - t0,
                )

                # v0.6.14+ 修复：L1/L3 模式下 TE 在 CPU，需要 patch encoders.py 的设备参数
                # 避免 input_ids 在 GPU 而 text_encoder.model 在 CPU 导致设备冲突
                #
                # L1：TE 一直在 CPU 编码（不临时上 GPU），patch 始终生效
                # L2：v0.6.16+ 改为 TE 一直在 CPU 编码（与 L1 一致）。
                #      之前 L2 让 TE 临时上 GPU 编码后 hook 回 CPU，但 hook 不可靠
                #      （SDNQ 量化后 te.model 调用链被修改），TE 留在 GPU → 扩散 OOM 崩溃。
                #      现统一用 patch：TE 在 CPU 编码，prompt_embeds 通过 patch 移到 GPU。
                #      代价：CPU 编码慢约 30s，但稳定不 OOM。
                # L3：TE 一直在 CPU 编码，patch 始终生效
                if self._offload_level in ("L1", "L2", "L3"):
                    self._patch_prompt_embeds_device()

                # ----- 7. v0.6+ 注入加速方案 -----
                # 必须在 offload 策略应用后注入（BlockSwap 依赖 transformer 已就位）
                t0 = time.time()
                try:
                    from .accelerators import apply_acceleration
                    self._acceleration_ctx = apply_acceleration(
                        pipe=self._pipe,
                        scheme=scheme,
                        logger=self._logger,
                    )
                    self._logger.info(
                        "加速方案注入完成: scheme=%s, 耗时=%.2fs",
                        scheme.scheme, time.time() - t0,
                    )
                except Exception as _accel_e:
                    # 加速方案注入失败不致命（降级到原生 attention），但记录警告
                    self._logger.warning(
                        "加速方案注入失败（降级到无加速）: %s", _accel_e,
                    )
                    self._acceleration_ctx = None

                self._loaded_variant = variant_name
                load_time = time.time() - start_time
                self._logger.info(
                    "MiniMax-H3 模型加载成功: variant=%s, scheme=%s, device=%s, offload=%s, total_time=%.2fs",
                    variant_name, scheme.scheme, self._device, self._offload_level, load_time,
                )

            except Exception as e:
                # 加载失败：清理已加载的部分
                self._pipe = None
                self._loaded_variant = None
                self._loaded_scheme_id = None
                self._acceleration_ctx = None
                err_str = str(e)
                if "CUDA out of memory" in err_str or "显存不足" in err_str:
                    raise RuntimeError(
                        f"GPU 显存不足，无法加载方案 {scheme.scheme}（需要 {scheme.vram_required}）。"
                        "建议：1) 关闭其他占用显存的程序 2) 切换到更低的方案（如 A→B→C）"
                        " 3) 卸载其他已加载的引擎"
                    ) from e
                raise RuntimeError(f"MiniMax-H3 模型加载失败: {e}") from e

    def _detect_paradigm(self, variant_name: str) -> str:
        """根据 variant 名识别范式（FL2VA / Ref2VA）。

        Args:
            variant_name: variant 名称

        Returns:
            "FL2VA" 或 "Ref2VA"

        Raises:
            ValueError: 无法识别范式
        """
        name_upper = variant_name.upper()
        for tag in _FL2VA_VARIANTS:
            if tag.upper() in name_upper:
                return "FL2VA"
        for tag in _REF2VA_VARIANTS:
            if tag.upper() in name_upper:
                return "Ref2VA"
        raise ValueError(
            f"无法识别 variant 范式（FL2VA / Ref2VA）: {variant_name}"
        )

    def _check_files_integrity(
        self,
        variant_name: str,
        diffusers_dir: Path,
        paradigm: str,
        scheme: Optional[SchemeConfig] = None,
    ) -> None:
        """校验模型目录下的关键文件是否齐全（不下载，只校验）。

        遵循项目规则「load() 中禁止兜底下载」：文件缺失时直接抛 RuntimeError，
        指引用户到模型管理界面下载。

        Args:
            variant_name: variant 名称
            diffusers_dir: 模型权重目录（SDNQ 4bit 或 BF16 原版）
            paradigm: 范式（FL2VA / Ref2VA）
            scheme: v0.6+ 方案配置（用于生成精确的下载指引）

        Raises:
            RuntimeError: 文件缺失（含缺失文件列表 + 下载指引）
        """
        # 关键配置文件（必需）
        # v0.6+ 修正：SDNQ 仓库无 model_index.json（只有 modular_model_index.json）
        # BF16 原版仓库两者都有
        required_paths = [
            "modular_model_index.json",  # SDNQ 和 BF16 都有此文件
            "text_encoder/config.json",
            "vae/config.json",
            "audio_vae/config.json",
            "scheduler/scheduler_config.json",
            "audio_scheduler/scheduler_config.json",
            "processor/tokenizer.json",
            "tokenizer/tokenizer.json",
        ]
        # BF16 原版额外需要 model_index.json
        # v0.7+：sdnq_uint4_fp16 与 sdnq_uint4 同为 SDNQ 仓库布局（无 model_index.json）
        if scheme and scheme.precision not in ("sdnq_uint4", "sdnq_uint4_fp16"):
            required_paths.append("model_index.json")

        # 根据范式选择 transformer
        if paradigm == "FL2VA":
            required_paths.append("transformer/config.json")
            required_paths.append("transformer/diffusion_pytorch_model.safetensors.index.json")
        else:  # Ref2VA
            required_paths.append("transformer_ref/config.json")
            required_paths.append("transformer_ref/diffusion_pytorch_model.safetensors.index.json")

        missing: list[str] = []
        for rel_path in required_paths:
            full_path = diffusers_dir / rel_path
            if not full_path.exists():
                missing.append(str(full_path))
            else:
                # 仅对非 json 文件校验大小（json 配置文件如 scheduler_config.json
                # 可能只有 90+ 字节，但仍为有效配置；权重文件小于 100B 才视为残文件）
                if not rel_path.endswith(".json"):
                    size = full_path.stat().st_size
                    if size < 100:
                        missing.append(f"{full_path} (size={size}B, 疑似残文件)")

        if missing:
            # v0.6+：从 scheme 生成精确下载指引
            repo_id = scheme.model_repo if scheme else "MiniMaxAI/MiniMax-H3"
            scheme_hint = (
                f"方案 {scheme.scheme}（precision={scheme.precision}）"
                if scheme else "未指定方案"
            )
            raise RuntimeError(
                f"模型文件不完整（{scheme_hint}），缺失 {len(missing)} 个文件：\n"
                f"  {missing}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，选择镜像源后下载。\n"
                f"HuggingFace 仓库：{repo_id}\n"
                f"本地路径：{diffusers_dir}\n"
                f"或手动下载命令：\n"
                f"  hf download {repo_id} --local-dir {diffusers_dir} \\\n"
                f"    --include \"transformer/*\" --include \"transformer_ref/*\" \\\n"
                f"    --include \"text_encoder/*\" --include \"vae/*\" --include \"audio_vae/*\" \\\n"
                f"    --include \"scheduler/*\" --include \"audio_scheduler/*\" \\\n"
                f"    --include \"processor/*\" --include \"tokenizer/*\" \\\n"
                f"    --include \"model_index.json\" --include \"modular_model_index.json\""
            )

        self._logger.info(
            "模型文件校验通过: variant=%s, paradigm=%s, files=%d, scheme=%s",
            variant_name, paradigm, len(required_paths),
            scheme.scheme if scheme else "N/A",
        )

    def _resolve_activation_dtype(self, precision: str) -> Any:
        """v0.7+ 按方案 precision 与 GPU 能力解析激活 dtype（Turing fp16 适配）。

        SDNQ 量化是 W4A16 形态：权重 uint4 不变，本方法决定的是激活/计算精度。

        规则：
            - "sdnq_uint4_fp16" / "bf16_fp16" → 显式 fp16（T 系 turbo 方案用，
              FP16 Tensor Core 从 sm_70 起可用）
            - "fp32" → fp32
            - "bf16" / "sdnq_uint4" → 默认 bf16；但当前 GPU 无 BF16 Tensor
              Core（compute capability < (8, 0)，如 Quadro RTX 8000 的
              Turing sm_75）时自动降级 fp16 并 WARNING

        关键：判断必须用 ``get_device_capability() >= (8, 0)``，不能用
        ``torch.cuda.is_bf16_supported()`` —— 后者在 Turing 上也返回 True
        （仅为慢速 CUDA core fallback kernel，GEMM 无 Tensor Core 路径，
        慢 10-50 倍），会漏降级（2026-08-26 在 Quadro RTX 8000 实测确认）。

        Args:
            precision: 方案精度标签（scheme.precision）

        Returns:
            torch.dtype（fp16 / bf16 / fp32）
        """
        import torch

        if precision in ("sdnq_uint4_fp16", "bf16_fp16"):
            self._logger.info("激活精度: fp16（显式档位 precision=%s）", precision)
            return torch.float16
        if precision == "fp32":
            return torch.float32

        # 默认 bf16 路径（bf16 / sdnq_uint4 / 未知值兜底）
        if precision in ("bf16", "sdnq_uint4") and torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability()
                if (major, minor) < (8, 0):
                    self._logger.warning(
                        "GPU %s (sm_%d%d) 无 BF16 Tensor Core（需 sm_80+），"
                        "bf16 激活自动降级为 fp16（bf16 在此 GPU 上走 CUDA core "
                        "模拟，慢 10-50 倍）: precision=%s",
                        torch.cuda.get_device_name(), major, minor, precision,
                    )
                    return torch.float16
            except Exception:
                pass
        return torch.bfloat16

    def _align_non_sdnq_dtype(self, component_module: Any, dtype: Any) -> int:
        """把组件内所有 float 参数/缓冲区统一到目标 dtype（v0.7+ fp16 适配）。

        根因 1（非量化层）：sdnq.load_sdnq_model(dtype=fp16) 内部
        apply_sdnq_options_to_module 只改写量化层 result_dtype → fp16；
        非量化层（embed_tokens / rmsnorm / lm_head 等）因 nn.Module 无
        `.dtype` property，被其 `getattr(model, "dtype", fp32) not in
        {fp32, fp64}` 条件跳过，权重保持 bf16 → embedding 输出 bf16
        与量化层 fp16 权重在 F.linear 处报 "BFloat16 != Half"。

        根因 2（SDNQ 层内部，v0.7.1 修复）：SDNQ 层的 scale / zero_point /
        svd_up / svd_down / bias 等浮点参数仍为原始 bf16。
        diffusers 的 AdaLN 调制层（transformer_minimax_h3.py
        MiniMaxH3AdaLayerNormModulation.forward）用
        `get_parameter_dtype(self.linear)` 决定激活转换目标 dtype——
        该函数返回层内**第一个参数**的 dtype（即 scale 的 bf16），
        导致输入被转成 bf16，与反量化后的 fp16 权重在
        quantized_linear_forward 的 matmul 处再次报
        "self and mat2 must have the same dtype, but got BFloat16 and Half"。

        v0.7.2 修复（全黑视频根因）：SDNQ 层 fp16 转换溢出。
        ``torch._int_mm`` 产生 INT32 结果（范围可达 ±1e7），
        直接转 fp16 会溢出（fp16 最大值 65504）→ Inf → scale 乘法后 NaN →
        latent NaN → VAE 解码全 0 → 全黑视频。
        修复：SDNQ 层的 scale / zero_point / svd / bias 统一转入 fp32
        （而非 fp16），确保 ``int_scaled_mm_torch`` 中 INT32→fp32 不溢出、
        scale 乘法 fp32 精度安全。非量化层保持 fp16 不变。

        Returns:
            转换的参数/buffer 数量（用于日志）
        """
        import torch
        from sdnq.layers import SDNQLayer

        # SDNQ 层内部的浮点参数（scale/zero_point/svd/bias）保持原 dtype（fp16），
        # 不做特殊处理。非量化层统一转 dtype。
        # INT32→fp16 溢出问题由 load() 中的 monkey-patch 解决。
        converted = 0
        for _, mod in component_module.named_modules():
            if isinstance(mod, SDNQLayer):
                # SDNQ 层：浮点参数保持原 dtype（fp16），
                # 整型打包权重（packed int weight）不动
                for _, param in list(mod.named_parameters(recurse=False)):
                    if param.is_floating_point() and param.dtype != dtype:
                        param.data = param.data.to(dtype)
                        converted += 1
                for bname, buf in list(mod.named_buffers(recurse=False)):
                    if buf is not None and buf.is_floating_point() and buf.dtype != dtype:
                        mod.register_buffer(bname, buf.to(dtype), persistent=False)
                        converted += 1
                continue
            for _, param in list(mod.named_parameters(recurse=False)):
                if param.is_floating_point() and param.dtype != dtype:
                    param.data = param.data.to(dtype)
                    converted += 1
            for bname, buf in list(mod.named_buffers(recurse=False)):
                if buf is not None and buf.is_floating_point() and buf.dtype != dtype:
                    mod.register_buffer(bname, buf.to(dtype), persistent=False)
                    converted += 1
        return converted

    def _load_sdnq_component(
        self,
        diffusers_dir: Path,
        subdir: str,
        attr: str,
        dtype: Any,
        model_cls: Any = None,
    ) -> None:
        """通过 sdnq.load_sdnq_model 加载 SDNQ 4bit 量化的组件（通用版）。

        v0.6.2+ 重构：从 _load_sdnq_transformer 改为通用方法，
        支持 transformer / transformer_ref / text_encoder 等任意 SDNQ 量化组件。

        v0.6.1+ 新增：解决 diffusers 不识别 quant_method=sdnq 的问题。
        详见 ANALYSIS-TASKS/H3_SDNQ_LOADER_INTEGRATION_ANALYSIS.md

        v0.6.2+ 修正：text_encoder 也使用 SDNQ 量化，必须通过 sdnq 库加载，
        否则 diffusers 会跳过量化处理，将 uint4 打包权重按 BF16 读入，
        导致内存访问违规崩溃。

        流程：
            1. 根据 subdir 自动推断 model_cls（若未显式传入）
               - transformer / transformer_ref → MiniMaxH3Transformer3DModel
               - text_encoder → Qwen3VLForConditionalGeneration
            2. 用 sdnq.load_sdnq_model 加载子目录
               （内部会读 config.json + quantization_config，应用 sdnq_post_load_quant）
            3. 通过 setattr 将组件注入 self._pipe
            4. 校验注入成功且带有 sdnq 量化标记

        Args:
            diffusers_dir: 模型根目录（如 .../sdnq_format）
            subdir: 子目录名（如 "transformer" / "transformer_ref" / "text_encoder"）
            attr: pipeline 上的属性名（通常与 subdir 相同）
            dtype: 加载精度（torch.bfloat16）
            model_cls: 模型类（None 则根据 subdir 自动推断）

        Raises:
            RuntimeError: 加载或注入失败
        """
        import torch

        # 自动推断 model_cls
        if model_cls is None:
            if subdir in ("transformer", "transformer_ref"):
                from diffusers import MiniMaxH3Transformer3DModel
                model_cls = MiniMaxH3Transformer3DModel
            elif subdir == "text_encoder":
                # text_encoder 是 Qwen3VL，用 transformers 的 AutoModel 或具体类
                # 注意：Qwen3VLForConditionalGeneration 在 transformers 5.13+ 可用
                try:
                    from transformers import Qwen3VLForConditionalGeneration
                    model_cls = Qwen3VLForConditionalGeneration
                except ImportError:
                    # 回退到 AutoModel
                    from transformers import AutoModel
                    model_cls = AutoModel
            else:
                # 通用回退：AutoModel
                from transformers import AutoModel
                model_cls = AutoModel

        component_path = diffusers_dir / subdir
        if not component_path.exists():
            raise RuntimeError(
                f"SDNQ 组件目录不存在: {component_path}\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮下载。"
            )

        # v0.6.5+ 缓存机制：同一 variant 重载时复用组件，跳过 sdnq.load_sdnq_model
        # 注：此前认为 sdnq 库第二次调用会 C 层 segfault，实际是 RAM 耗尽导致 OOM。
        # v0.7+ 缓存 key 扩展为 subdir:dtype —— 切换方案（如 A→T8，bf16→fp16）
        # 时 dtype 不同，不能复用旧 dtype 缓存组件，否则量化层 result_dtype
        # 与实际加载请求不一致。
        # 卸载时清空缓存（_unload_unsafe），避免切换 variant 时 RAM 叠加 OOM。
        cache_key = f"{subdir}:{dtype}"
        if cache_key in self._cached_sdnq_components:
            cached = self._cached_sdnq_components[cache_key]
            self._logger.info(
                "SDNQ 组件复用缓存: subdir=%s, dtype=%s, type=%s（跳过 sdnq.load_sdnq_model）",
                subdir, dtype, type(cached).__name__,
            )
            setattr(self._pipe, attr, cached)
            return

        self._logger.info(
            "sdnq.load_sdnq_model: path=%s, model_cls=%s, dtype=%s",
            component_path, getattr(model_cls, "__name__", str(model_cls)), dtype,
        )

        try:
            from sdnq import load_sdnq_model
        except ImportError as _ie:
            raise RuntimeError(
                "sdnq 库未安装，无法加载 SDNQ 4bit 量化模型。\n"
                "请在 envs/video-minimax-h3 环境中执行: pip install sdnq\n"
                f"原始错误: {_ie}"
            ) from _ie

        # v0.6.3+ 修复 sdnq 库与 safetensors 0.8+ 不兼容问题：
        # sdnq 0.2.4 的 file_loader.load_safetensors 使用 `for key in f:` 旧写法，
        # 但 safetensors 0.8+ 的 safe_open 对象不再支持 __iter__，必须用 f.keys()。
        # 错误：TypeError: 'builtins.safe_open' object is not iterable
        # 解决：monkey-patch sdnq.file_loader.load_safetensors 用 f.keys() 替代迭代。
        try:
            import sdnq.file_loader as _sdnq_file_loader
            import torch as _torch_for_patch

            if not getattr(_sdnq_file_loader, "_aireliable_patched", False):
                _orig_map_keys = _sdnq_file_loader.map_keys

                def _patched_load_safetensors(files, state_dict=None, key_mapping=None, device=_torch_for_patch.device("cpu")):
                    from safetensors.torch import safe_open
                    if state_dict is None:
                        state_dict = {}
                    for fn in files:
                        with safe_open(fn, framework="pt", device=str(device)) as f:
                            for key in f.keys():  # 修复：用 f.keys() 替代 for key in f
                                state_dict[_orig_map_keys(key, key_mapping)] = f.get_tensor(key)

                _sdnq_file_loader.load_safetensors = _patched_load_safetensors
                _sdnq_file_loader._aireliable_patched = True
                self._logger.debug("已 monkey-patch sdnq.file_loader.load_safetensors（适配 safetensors 0.8+）")
        except Exception as _patch_e:
            self._logger.warning("monkey-patch sdnq.file_loader 失败: %s", _patch_e)

        # v0.6.3+ 修复 sdnq 库 UnboundLocalError：
        # sdnq.loader.load_sdnq_model 中 `import transformers` 仅在 `if model_cls is None:`
        # 块内执行，但 Python 会把 transformers 当作整个函数的局部变量。
        # 因此当 model_cls 不为 None 时，elif 分支调用 transformers.AutoConfig 会 UnboundLocalError。
        # 解决：对 text_encoder 传 model_cls=None，让 sdnq 走 if 块触发 import transformers，
        # sdnq 会从 config.json 的 architectures 字段自动推断为 Qwen3VLForConditionalGeneration。
        # transformer/transformer_ref 是 diffusers 类，走 from_config 路径，不受此 bug 影响。
        sdnq_model_cls = model_cls
        if subdir == "text_encoder":
            sdnq_model_cls = None
            self._logger.info(
                "text_encoder 强制传 model_cls=None 给 sdnq（让 sdnq 自动推断，规避 UnboundLocalError）",
            )

        try:
            # v0.6.11+ CPU Offload 策略：根据 offload_level 和组件类型决定加载设备
            # 解决 48GB 显存被模型权重占满导致推理 OOM 的问题。
            #
            # 策略矩阵：
            #   L0（全部 GPU）   : text_encoder → GPU, transformer → GPU
            #   L1/L2（轻度/中度）: text_encoder → CPU（用 RAM），transformer → GPU
            #                      （text_encoder 仅在 prompt 编码时需要，可在 CPU 上运行）
            #   L3（重度卸载）    : text_encoder → CPU，transformer → CPU
            #                      （推理时按需把 transformer 移到 GPU）
            #
            # 关键约束：SDNQ 4bit 权重必须在 load_sdnq_model 时指定最终设备，
            # 后续 .to(cuda) 可能导致 dequant 失败。所以根据 offload 等级
            # 在加载阶段就决定设备。
            sdnq_device = self._resolve_sdnq_load_device(subdir)
            self._logger.info(
                "SDNQ 加载设备决策: subdir=%s, offload=%s, device=%s",
                subdir, self._offload_level, sdnq_device,
            )
            component_module = load_sdnq_model(
                model_path=str(component_path),
                model_cls=sdnq_model_cls,
                dtype=dtype,
                device=sdnq_device,
            )
        except Exception as _e:
            self._logger.exception("sdnq.load_sdnq_model 调用失败 (subdir=%s)", subdir)
            raise RuntimeError(
                f"SDNQ 组件加载失败 (subdir={subdir}): {_e}\n"
                f"路径: {component_path}\n"
                f"可能原因：\n"
                f"  1. 权重文件损坏（请重新下载）\n"
                f"  2. sdnq 库版本与权重不兼容（当前: {__import__('sdnq').__version__}）\n"
                f"  3. config.json 的 quantization_config 配置错误"
            ) from _e

        # v0.7+ fp16 适配：非 SDNQ 层（embedding/norm 等）dtype 对齐
        # sdnq loader 只改写量化层 result_dtype，非量化层保持 bf16 会导致
        # F.linear 处 "BFloat16 != Half"（详见 _align_non_sdnq_dtype docstring）
        if dtype is not None:
            aligned = self._align_non_sdnq_dtype(component_module, dtype)
            self._logger.info(
                "SDNQ 组件非量化层 dtype 对齐: subdir=%s, target_dtype=%s, 转换参数/buffer 数=%d",
                subdir, dtype, aligned,
            )

        # 注入到 pipeline
        setattr(self._pipe, attr, component_module)
        self._logger.info(
            "SDNQ 组件已注入 pipeline: attr=%s, type=%s",
            attr, type(component_module).__name__,
        )

        # v0.6.5+ 缓存组件（同一 variant 重载时复用，跳过 sdnq.load_sdnq_model）
        self._cached_sdnq_components[cache_key] = component_module
        self._logger.info(
            "SDNQ 组件已缓存: subdir=%s（同一 variant 重载时复用）",
            subdir,
        )

        # 校验量化标记（防止 sdnq 静默失败导致后续推理输出乱码）
        has_quant_config = (
            hasattr(component_module, "quantization_config")
            or hasattr(component_module, "hf_quantizer")
            or any(
                hasattr(m, "sdnq_dequantizer")
                for m in component_module.modules()
            )
        )
        if not has_quant_config:
            self._logger.warning(
                "SDNQ 组件加载完成但未检测到量化标记 (attr=%s)，"
                "可能 sdnq_post_load_quant 未生效（推理结果可能异常）",
                attr,
            )
        else:
            self._logger.info("SDNQ 量化标记校验通过 (attr=%s)", attr)

        # v0.6.7+ 修复 SDNQ transformer forward 签名问题：
        # diffusers denoise.py 用 `inspect.signature(transformer.forward).parameters` 过滤
        # layout_kwargs（token_tags/position_ids/video_indices/audio_indices/text_indices），
        # 但 SDNQ 包装后的 forward 签名可能缺少这些参数，导致它们被过滤掉，
        # transformer forward 抛 "missing 5 required positional arguments" 错误。
        # 解决：给 SDNQ transformer.forward 包装一层，设置 __signature__ 为原始 forward 签名。
        if subdir in ("transformer", "transformer_ref"):
            self._patch_sdnq_forward_signature(component_module, subdir)

    def _patch_sdnq_forward_signature(self, component_module: Any, subdir: str) -> None:
        """修复 SDNQ transformer forward 签名问题。

        diffusers 的 denoise.py (MiniMaxH3LoopDenoiser.__call__) 使用：
            inspect.signature(transformer.forward).parameters
        来过滤 denoiser_input_fields（token_tags/position_ids/video_indices/
        audio_indices/text_indices）。SDNQ 量化后的 forward 方法签名可能被
        包装/修改，导致 inspect.signature 无法正确获取这些参数，从而被过滤掉，
        引发 forward() missing required arguments 错误。

        v0.6.8+ 修复方案：直接 monkey-patch MiniMaxH3LoopDenoiser.__call__
        和 MiniMaxH3Ref2VALoopDenoiser.__call__，不再使用 inspect.signature
        过滤 denoiser_input_fields，而是直接传递所有字段。

        Args:
            component_module: SDNQ 加载的 transformer 实例
            subdir: 子目录名（transformer / transformer_ref）
        """
        import torch
        import inspect

        # 只需 patch 一次（全局生效）
        try:
            from diffusers.modular_pipelines.minimax_h3.denoise import (
                MiniMaxH3LoopDenoiser,
                MiniMaxH3Ref2VALoopDenoiser,
            )
        except ImportError:
            self._logger.warning("无法导入 MiniMaxH3LoopDenoiser，跳过 denoise patch")
            return

        # 检查是否已 patch（避免重复 patch）
        if getattr(MiniMaxH3LoopDenoiser, "_aireliable_patched", False):
            self._logger.info("MiniMaxH3LoopDenoiser.__call__ 已 patch（跳过）")
            return

        # 保存原始 __call__ 方法
        _orig_call = MiniMaxH3LoopDenoiser.__call__

        @torch.no_grad()
        def _patched_call(self, components, block_state, i, t):
            """修复版 __call__：保留原版过滤逻辑，但强制包含 SDNQ 签名丢失的必需参数。

            背景：
            - 原版用 inspect.signature(transformer.forward).parameters 过滤 denoiser_input_fields
            - SDNQ 量化后 transformer.forward 签名被修改，inspect.signature 报告缺少
              token_tags/position_ids/video_indices/audio_indices/text_indices 等参数
            - 这些参数 transformer 实际接受（运行时正常），只是签名信息丢失
            - 同时 denoiser_input_fields 可能包含 prompt_embeds 等 transformer 不接受的参数，
              原版过滤会移除它们，不能简单传递全部

            修复策略：
            1. 保留原版 inspect.signature 过滤（移除 transformer 不接受的参数，如 prompt_embeds）
            2. 强制包含已知 SDNQ 签名丢失但 transformer 实际接受的参数
            """
            transformer = getattr(components, self.transformer_name)
            unique_timesteps, timestep_indices = block_state.row_timestep_plan[i]
            # v0.6.11+ 修复：SDNQ 量化后 rope.inv_freq buffer 设备不一致
            # inv_freq 是 persistent=False buffer，SDNQ 量化 .to(device) 可能跳过它
            # 每次调用 transformer 前都检查并修复 rope 的所有 buffer 设备
            try:
                if hasattr(transformer, "rope"):
                    rope = transformer.rope
                    target_device = next(transformer.parameters()).device
                    # 直接强制移动 inv_freq 到正确设备（不检查，直接 to）
                    if hasattr(rope, "inv_freq"):
                        if rope.inv_freq.device != target_device:
                            # 使用 .data = .data.to() 避免触发 hook
                            rope.inv_freq.data = rope.inv_freq.data.to(target_device)
                    # 检查 rope 的所有 buffer
                    for buf_name, buf in rope.named_buffers():
                        if buf.device != target_device:
                            buf.data = buf.data.to(target_device)
            except Exception:
                pass  # 静默失败，让 transformer 自己抛出更详细的错误
            # SDNQ 量化后签名丢失但 transformer 实际接受的参数
            required_layout_fields = {
                "token_tags",
                "position_ids",
                "video_indices",
                "audio_indices",
                "text_indices",
            }
            sig_params = inspect.signature(transformer.forward).parameters
            layout_kwargs = {
                name: value
                for name, value in block_state.denoiser_input_fields.items()
                if name in sig_params or name in required_layout_fields
            }
            # v0.6.13+ 修复：L2 模式下 TE 在 CPU 编码，prompt_embeds 可能在 CPU
            # 强制把 prompt_embeds 移到 transformer 的设备（避免设备冲突）
            _target_device = next(transformer.parameters()).device
            _prompt_embeds = block_state.prompt_embeds
            if hasattr(_prompt_embeds, "device") and _prompt_embeds.device != _target_device:
                _prompt_embeds = _prompt_embeds.to(_target_device)
                block_state.prompt_embeds = _prompt_embeds  # 更新供后续 step 复用
            # v0.6.15+ 修复：scheduler.step 返回的 latents/audio_latents 可能在 CPU
            # （L2 模式下 VAE 临时下 CPU 后，scheduler 计算可能产生 CPU 张量）
            # 强制把 latents/audio_latents 移到 transformer 的设备，避免 hidden_states 在 CPU
            for _field_name in ("latents", "audio_latents"):
                _field = getattr(block_state, _field_name, None)
                if hasattr(_field, "device") and _field.device != _target_device:
                    setattr(block_state, _field_name, _field.to(_target_device))
            # v0.6.17+ 修复：layout_kwargs 中的张量（text_indices/video_indices/audio_indices/
            # token_tags/position_ids）可能在 CPU，但 transformer 内部用 index_copy 与
            # hidden_states(cuda) 操作会报设备冲突。强制把所有张量参数移到 transformer 设备。
            for _name, _value in list(layout_kwargs.items()):
                if hasattr(_value, "device") and _value.device != _target_device:
                    layout_kwargs[_name] = _value.to(_target_device)
            # v0.6.19+ 修复：timestep / timestep_indices 在 CPU 时，transformer.forward 内部
            # self.time_proj(timestep) → self.time_embedder(temb.to(dtype))，.to(dtype) 只转
            # dtype 不转 device，导致 temb 在 CPU 而 time_embedder 参数在 GPU，触发
            # "mat1 is on cpu" 错误。强制把所有时间步相关张量移到 transformer 设备。
            if hasattr(unique_timesteps, "device") and unique_timesteps.device != _target_device:
                unique_timesteps = unique_timesteps.to(_target_device)
            if hasattr(timestep_indices, "device") and timestep_indices.device != _target_device:
                timestep_indices = timestep_indices.to(_target_device)
            # ---- v1.2 NaN 分步探针（诊断用，定位发散起点）----
            try:
                import torch as _pw_torch
                _in_lat = block_state.latents
                _pw_log = [
                    "i=%s" % i,
                    "lat_in_nan=%.6f" % (float(_pw_torch.isnan(_in_lat).float().mean()) if _in_lat is not None else -1.0),
                    "lat_in_inf=%.6f" % (float(_pw_torch.isinf(_in_lat).float().mean()) if _in_lat is not None else -1.0),
                    "mean=%.4g" % (float(_in_lat.float().mean()) if _in_lat is not None else -1.0),
                    "absmax=%.4g" % (float(_in_lat.float().abs().max()) if _in_lat is not None else -1.0),
                ]
                _logref = globals().get("logger", __import__("logging").getLogger("minimax_h3_wrapper"))
                _logref.warning("[NaN探针] 输入 %s", " ".join(_pw_log))
            except Exception as _pw_e:
                _logref = globals().get("logger", __import__("logging").getLogger("minimax_h3_wrapper"))
                try:
                    _logref.warning("[NaN探针] 输入探针异常: %s", _pw_e)
                except Exception:
                    pass
            block_state.noise_pred, block_state.audio_noise_pred = transformer(
                hidden_states=block_state.latents[None],
                audio_hidden_states=block_state.audio_latents[None],
                encoder_hidden_states=_prompt_embeds,
                timestep=unique_timesteps,
                timestep_indices=timestep_indices,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **layout_kwargs,
            )
            try:
                import torch as _pw_torch2
                _np = block_state.noise_pred
                _anp = block_state.audio_noise_pred
                _logref2 = globals().get("logger", __import__("logging").getLogger("minimax_h3_wrapper"))
                _logref2.warning(
                    "[NaN探针] 输出 noise_pred(nan=%.6f inf=%.6f mean=%.4g absmax=%.4g) audio_np(nan=%.6f inf=%.6f)",
                    float(_pw_torch2.isnan(_np).float().mean()), float(_pw_torch2.isinf(_np).float().mean()),
                    float(_np.float().mean()), float(_np.float().abs().max()),
                    (float(_pw_torch2.isnan(_anp).float().mean()) if _anp is not None else -1.0),
                    (float(_pw_torch2.isinf(_anp).float().mean()) if _anp is not None else -1.0),
                )
            except Exception as _pw_e2:
                _logref2 = globals().get("logger", __import__("logging").getLogger("minimax_h3_wrapper"))
                try:
                    _logref2.warning("[NaN探针] 输出探针异常: %s", _pw_e2)
                except Exception:
                    pass
            return components, block_state

        # 应用 patch
        MiniMaxH3LoopDenoiser.__call__ = _patched_call
        MiniMaxH3LoopDenoiser._aireliable_patched = True

        # v0.6.11+ 修复：Monkey-patch MiniMaxH3RotaryPosEmbed.forward
        # 解决 SDNQ 量化后 inv_freq buffer 设备不一致问题
        # inv_freq 是 persistent=False buffer，SDNQ 量化的 .to(device) 可能跳过它
        # 在 rope.forward 中强制把 inv_freq 移到 position_ids 的设备
        try:
            from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
            if not getattr(MiniMaxH3RotaryPosEmbed, "_aireliable_rope_patched", False):
                _orig_rope_forward = MiniMaxH3RotaryPosEmbed.forward

                def _patched_rope_forward(self, position_ids):
                    # 强制把 inv_freq 移到 position_ids 的设备
                    if hasattr(self, "inv_freq"):
                        target_device = position_ids.device
                        if self.inv_freq.device != target_device:
                            self.inv_freq.data = self.inv_freq.data.to(target_device)
                    return _orig_rope_forward(self, position_ids)

                MiniMaxH3RotaryPosEmbed.forward = _patched_rope_forward
                MiniMaxH3RotaryPosEmbed._aireliable_rope_patched = True
                self._logger.info("MiniMaxH3RotaryPosEmbed.forward 已 patch（修复 inv_freq 设备冲突）")
        except Exception as _rope_patch_e:
            self._logger.warning("MiniMaxH3RotaryPosEmbed.forward patch 失败: %s", _rope_patch_e)

        # ===== v1.3.1 定位+修复 NaN：transformer 前向内部阶段探针 + 注意力 fp32 =====
        # 背景：权重探针已确认 362 个 sdnq 线性层去量化权重无 NaN/Inf，SDNQ scaled-MM 已 patch 为 fp32，
        #      但 Step0 输出 noise_pred 仍 100% NaN。唯一未 patch 的裸 fp16 计算是 attention 的 QK@softmax@V，
        #      （q/k 逐元素乘积在 fp16 超过 ±65504 → Inf → softmax NaN → 残差级联全 NaN）。
        # 措施 A：把 MiniMaxH3AttnProcessor.__call__ 的 QKV 计算提升到 fp32，软max/attention 不再溢出。
        # 措施 B：在 MiniMaxH3Transformer3DModel.forward 埋阶段探针，精确定位 NaN 首发阶段。
        try:
            from diffusers.models.transformers.transformer_minimax_h3 import (
                MiniMaxH3Transformer3DModel,
                MiniMaxH3TransformerOutput,
                MiniMaxH3AttnProcessor,
                _apply_rotary_emb,
            )
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            # ---- 措施 A：注意力 QKV+attention 提升到 fp32 ----
            if not getattr(MiniMaxH3AttnProcessor, "_aireliable_attn_fp32_patched", False):
                _orig_attn_call = MiniMaxH3AttnProcessor.__call__

                def _patched_attn_call(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
                    # Q/K/V 投影在模型原生 dtype（fp16/SDNQ 层）下计算，避免 F.linear(fp32, 权重fp16) dtype 不匹配；
                    # 仅把注意力打分（Q@K@V 以及 softmax）提升到 fp32，杜绝 fp16 溢出导致的 NaN。
                    if attn.fused_projections:
                        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
                    else:
                        query = attn.to_q(hidden_states)
                        key = attn.to_k(hidden_states)
                        value = attn.to_v(hidden_states)
                    query = query.unflatten(-1, (attn.heads, -1))
                    key = key.unflatten(-1, (attn.heads, -1))
                    value = value.unflatten(-1, (attn.heads, -1))
                    if getattr(attn, "norm_q", None) is not None:
                        query = attn.norm_q(query)
                    if getattr(attn, "norm_k", None) is not None:
                        key = attn.norm_k(key)
                    if rotary_emb is not None:
                        query = _apply_rotary_emb(query, *rotary_emb)
                        key = _apply_rotary_emb(key, *rotary_emb)
                    _act_dtype = value.dtype  # 投影输出 dtype（fp16），attention 结果转回该 dtype
                    query = query.float()
                    key = key.float()
                    value = value.float()
                    hidden_states = dispatch_attention_fn(
                        query, key, value,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                        backend=self._attention_backend,
                        parallel_config=self._parallel_config,
                    )
                    hidden_states = hidden_states.flatten(2, 3).to(_act_dtype)
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)
                    return hidden_states

                MiniMaxH3AttnProcessor.__call__ = _patched_attn_call
                MiniMaxH3AttnProcessor._aireliable_attn_fp32_patched = True
                self._logger.info("MiniMaxH3AttnProcessor.__call__ 已 patch（QKV/attention 提升到 fp32 防溢出）")

            # ---- 措施 B：transformer forward 阶段探针 ----
            if not getattr(MiniMaxH3Transformer3DModel, "_aireliable_tf_probe_patched", False):
                MINIMAX_H3_MODALITY_NUM = 3

                def _patched_tf_forward(
                        self, hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
                        timestep_indices, token_tags, position_ids, video_indices, audio_indices, text_indices,
                        attention_kwargs=None, return_dict=True):
                    from diffusers.models.modeling_utils import get_parameter_dtype
                    _logr = globals().get("logger", __import__("logging").getLogger("minimax_h3_wrapper"))
                    _dbg = not getattr(self, "_aireliable_tf_dbg_done", False)

                    def _np(t, tag):
                        try:
                            if t is None or getattr(t, "numel", lambda: 0)() == 0:
                                return
                            _logr.warning("[TF探针] %s: nan=%.4f inf=%.4f absmax=%.4f dtype=%s shape=%s",
                                          tag, float(t.isnan().float().mean()),
                                          float(t.isinf().float().mean()), float(t.abs().max()),
                                          t.dtype, list(t.shape))
                        except Exception:
                            pass

                    # ---- v1.4.0 根因修复：transformer 整体 fp32 计算 ----
                    # 背景：MiniMax-H3 参考实现是 BF16（范围 ±3e38），SDNQ uint4_fp16 强制 fp16（±65504），
                    #       激活值合法可到 ~2.5e4+，fp16 下 FeedForward 与 gated residual 相加溢出 -> Inf -> NaN 灰屏。
                    # 修复：把所有 SDNQ 线性层改为 fp32 全去量化 forward（权重仍是量化存储，瞬态去量化 fp32，
                    #       内存可控，transformer SDNQ 层共 362 个），非 SDNQ 小模块 float；阻断 fp16 溢出。
                    if not getattr(self, "_aireliable_fp32_done", False):
                        self._aireliable_fp32_done = True
                        import types as _types
                        _fp32_n = 0
                        for _m in self.modules():
                            if getattr(_m, "sdnq_dequantizer", None) is not None:
                                def _sd_fwd(_s, _input):
                                    _w = _s.sdnq_dequantizer(_s.weight, _s.scale, _s.zero_point, _s.svd_up, _s.svd_down)
                                    _b = _s.bias
                                    return torch.nn.functional.linear(
                                        _input.to(torch.float32), _w.to(torch.float32),
                                        _b.to(torch.float32) if _b is not None else None,
                                    )
                                try:
                                    _m.forward = _types.MethodType(_sd_fwd, _m)
                                    _fp32_n += 1
                                except Exception:
                                    pass
                            else:
                                if isinstance(_m, (torch.nn.RMSNorm, torch.nn.Linear)):
                                    try:
                                        _m.to(torch.float32)
                                    except Exception:
                                        pass
                        _logr.warning("[TF探针] transformer 已整体转为 fp32 计算（SDNQ 层 fp32 全去量化，共 %d 层）", _fp32_n)

                    hidden_states = hidden_states.to(torch.float32)
                    audio_hidden_states = audio_hidden_states.to(torch.float32)
                    encoder_hidden_states = encoder_hidden_states.to(torch.float32)

                    rotary_emb = self.rope(position_ids)
                    video_embeds = self.proj_in(hidden_states)
                    audio_embeds = self.audio_proj_in(audio_hidden_states)
                    text_embeds_pre = self.context_embedder(encoder_hidden_states)
                    text_embeds = self.token_refiner(text_embeds_pre)

                    packed = text_embeds.new_zeros((text_embeds.shape[0], position_ids.shape[0], text_embeds.shape[-1]))
                    packed = packed.index_copy(1, text_indices, text_embeds)
                    packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
                    packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

                    temb = self.time_proj(timestep)
                    temb = self.time_embedder(temb.to(torch.float32))
                    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

                    hs = packed
                    if _dbg:
                        _np(video_embeds, "video_embeds")
                        _np(encoder_hidden_states, "text输入TE")
                        _np(text_embeds_pre, "text_context(embedder后)")
                        _np(text_embeds, "text_refined")
                        _np(packed, "packed序列")
                        _np(temb, "temb")
                    for bi, block in enumerate(self.transformer_blocks):
                        hs = block(hs, temb, adaln_indices, rotary_emb)
                        if _dbg and bi < 3:
                            _np(hs, f"block{bi}输出")

                    _out_hs = self.norm_out(hs, temb, timestep_indices)
                    # proj_out 权重已被整体 fp32 化，输入需保持 fp32，输出再转回 fp16
                    video_output = self.proj_out(_out_hs).index_select(1, video_indices).to(torch.float16)
                    audio_output = self.audio_proj_out(_out_hs).index_select(1, audio_indices).to(torch.float16)
                    if _dbg:
                        self._aireliable_tf_dbg_done = True
                        _np(_out_hs, "norm_out后")
                        _np(video_output, "video_output")

                    if not return_dict:
                        return (video_output, audio_output)
                    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

                MiniMaxH3Transformer3DModel.forward = _patched_tf_forward
                MiniMaxH3Transformer3DModel._aireliable_tf_probe_patched = True
                self._logger.info("MiniMaxH3Transformer3DModel.forward 阶段探针已 patch")

                # 在 block.forward 上做子阶段 NaN 定位（首个前向），精确定位 NaN 首发算术
                from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
                if not getattr(MiniMaxH3TransformerBlock, "_aireliable_blk_probe_patched", False):
                    _orig_blk_fwd = MiniMaxH3TransformerBlock.forward

                    def _patched_blk_fwd(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
                        _logr = globals().get("logger", __import__("logging").getLogger("minimax_h3_wrapper"))
                        _dbg = not getattr(self, "_aireliable_blk_dbg_done", False)

                        def _np(t, tag):
                            try:
                                if t is None or getattr(t, "numel", lambda: 0)() == 0:
                                    return
                                _logr.warning("[BLK探针] %s: nan=%.4f inf=%.4f absmax=%.4f dtype=%s",
                                              tag, float(t.isnan().float().mean()),
                                              float(t.isinf().float().mean()), float(t.abs().max()), t.dtype)
                            except Exception:
                                pass

                        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
                        residual = hidden_states
                        norm_hidden_states = self.norm1(hidden_states)
                        norm_hidden_states = norm_hidden_states * (
                            1.0 + scale_msa.index_select(0, adaln_indices)
                        ) + shift_msa.index_select(0, adaln_indices)
                        attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
                        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attn_output

                        residual = hidden_states
                        norm_hidden_states = self.norm2(hidden_states)
                        norm_hidden_states = norm_hidden_states * (
                            1.0 + scale_mlp.index_select(0, adaln_indices)
                        ) + shift_mlp.index_select(0, adaln_indices)
                        ff_output = self.ff(norm_hidden_states)
                        hidden_states = residual + gate_mlp.index_select(0, adaln_indices) * ff_output

                        if _dbg:
                            try:
                                self._aireliable_blk_dbg_done = True
                                _np(self.adaln_proj(temb)[0], "adaln_out0")
                                _np(hidden_states, "final")
                                # scale_msa/gate_msa 是否 NaN
                                t0 = self.adaln_proj(temb)
                                _np(t0[0], "shift_msa")
                                _np(t0[1], "scale_msa")
                                _np(t0[2], "gate_msa")
                                _np(norm_hidden_states, "norm2_gated")
                                _np(ff_output, "ff_output")
                            except Exception:
                                pass
                        return hidden_states

                    MiniMaxH3TransformerBlock.forward = _patched_blk_fwd
                    MiniMaxH3TransformerBlock._aireliable_blk_probe_patched = True
                    self._logger.info("MiniMaxH3TransformerBlock.forward 子阶段探针已 patch")

                # ---- 措施 C：token_refiner 块内探针（精确定位 text 路径 NaN 首发：attn vs swiglu FF）----
                # 背景：text_refined 在 block0 之前已 100% NaN。text 仅 22 tokens，refiner 的 attention
                #       走 fp32 补丁，但 swiglu FeedForward/norm 仍 fp16，疑似 swiglu 中间值溢出。
                try:
                    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TokenRefinerBlock
                    if not getattr(MiniMaxH3TokenRefinerBlock, "_aireliable_refiner_probe_patched", False):
                        _orig_refblock = MiniMaxH3TokenRefinerBlock.forward

                        def _patched_refblock(self, hidden_states):
                            _logr = globals().get("logger", __import__("logging").getLogger("minimax_h3_wrapper"))
                            _dbg = not getattr(self, "_aireliable_refiner_dbg_done", False)

                            def _np(t, tag):
                                try:
                                    if t is None or getattr(t, "numel", lambda: 0)() == 0:
                                        return
                                    _logr.warning("[REF探针] %s: nan=%.4f inf=%.4f absmax=%.4f dtype=%s",
                                                  tag, float(t.isnan().float().mean()),
                                                  float(t.isinf().float().mean()), float(t.abs().max()), t.dtype)
                                except Exception:
                                    pass

                            n1 = self.norm1(hidden_states)
                            attn_out = self.attn(n1)
                            a = hidden_states + attn_out
                            n2 = self.norm2(a)
                            ff_out = self.ff(n2)
                            out = a + ff_out
                            if _dbg:
                                self._aireliable_refiner_dbg_done = True
                                _np(hidden_states, "refiner输入")
                                _np(n1, "refiner_norm1")
                                _np(attn_out, "refiner_attn_out")
                                _np(a, "refiner_attn_residual")
                                _np(n2, "refiner_norm2")
                                _np(ff_out, "refiner_ff_out")
                            return out

                        MiniMaxH3TokenRefinerBlock.forward = _patched_refblock
                        MiniMaxH3TokenRefinerBlock._aireliable_refiner_probe_patched = True
                        self._logger.info("MiniMaxH3TokenRefinerBlock.forward 探针已 patch")
                except Exception as _refprobe_e:
                    self._logger.warning("token_refiner 探针 patch 失败: %s", _refprobe_e)
        except Exception as _tfprobe_e:
            self._logger.warning("transformer forward 探针/attention fp32 patch 失败: %s", _tfprobe_e)

        # MiniMaxH3Ref2VALoopDenoiser 继承自 MiniMaxH3LoopDenoiser，
        # 但它有自己的 __call__ 吗？不，它只覆盖了 __init__。
        # 所以 patch 父类即可。
        # 但为了保险，也检查 Ref2VA 是否需要单独 patch
        if MiniMaxH3Ref2VALoopDenoiser.__call__ is _orig_call:
            MiniMaxH3Ref2VALoopDenoiser.__call__ = _patched_call
            self._logger.info("MiniMaxH3Ref2VALoopDenoiser.__call__ 也已 patch")

        # 验证 patch
        try:
            sig = inspect.signature(_patched_call)
            self._logger.info(
                "MiniMaxH3LoopDenoiser.__call__ patch 成功（不再使用 inspect.signature 过滤 denoiser_input_fields）",
            )
            # 检查 SDNQ forward 签名（仅日志记录，不再依赖签名修复）
            try:
                sdnq_sig = inspect.signature(component_module.forward)
                sdnq_params = list(sdnq_sig.parameters.keys())
                required = ["token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"]
                missing = [p for p in required if p not in sdnq_params]
                if missing:
                    self._logger.warning(
                        "SDNQ %s.forward 签名仍缺少: %s（但已通过 denoise patch 绕过）",
                        subdir, missing,
                    )
                else:
                    self._logger.info(
                        "SDNQ %s.forward 签名包含所有必需参数（%d 参数）",
                        subdir, len(sdnq_params),
                    )
            except Exception:
                pass
        except Exception as _e:
            self._logger.error("MiniMaxH3LoopDenoiser.__call__ patch 验证失败: %s", _e)

    def _patch_prompt_embeds_device(self) -> None:
        """v0.6.16+ 修复：L2 模式下 TE 在 CPU，但 CPU 编码 Qwen3-VL（19.6GB）极慢（5分钟+）。

        新策略（函数级 TE 设备控制，不依赖 hook）：
        - L2 模式：TE 临时上 GPU 编码（~30s），编码完成后立即下 CPU（释放显存给 attention）
        - L1/L3 模式：TE 保持在 CPU 编码（用 patch 让 device=cpu 避免设备冲突）

        与之前的 hook 方案区别：
        - hook 注册在 te.model 上，但 SDNQ 量化后调用链被修改，hook 可能不触发 → TE 留在 GPU → OOM
        - 新方案直接在 get_qwen3vl_prompt_embeds 函数内控制 TE 设备，编码后立即下 CPU，100% 可靠
        """
        try:
            import torch
            from diffusers.modular_pipelines.minimax_h3 import encoders as encoders_module

            _orig_get_embeds = encoders_module.get_qwen3vl_prompt_embeds

            if getattr(_orig_get_embeds, "_aireliable_patched", False):
                self._logger.info("get_qwen3vl_prompt_embeds 已 patch（跳过）")
                return

            _offload_level = self._offload_level
            _logger = self._logger
            _device = self._device
            _get_gpu_mem_mb = self._get_gpu_memory_used_mb  # 闭包引用，供 patch 函数使用

            # v0.6.18+ 设置 pipe 引用到 text_encoder，供 patch 函数卸载/恢复 VAE
            if self._pipe is not None and hasattr(self._pipe, "text_encoder"):
                try:
                    self._pipe.text_encoder._aireliable_pipe_ref = self._pipe
                    _logger.info("[TE Patch] 已设置 _aireliable_pipe_ref 到 text_encoder")
                except Exception as _ref_e:
                    _logger.warning("[TE Patch] 设置 _aireliable_pipe_ref 失败: %s", _ref_e)

            def _patched_get_embeds(text_encoder, processor, token_ids, vision_inputs=None,
                                     text_encoder_layer=50, device=None, dtype=None):
                _te_moved_to_gpu = False
                _vae_offloaded = []
                _transformer_offloaded = []  # v0.7.3+ 长 token 编码时临时下 CPU 的 transformer 组件

                # v0.6.18+ 调试日志：确认 patch 被调用及各条件值
                try:
                    _te_dev = next(text_encoder.parameters()).device
                    _logger.info(
                        "[TE Patch 调试] _patched_get_embeds 被调用: offload=%s, _device=%s, "
                        "te_device=%s, device_arg=%s, token_ids_len=%s",
                        _offload_level, _device, _te_dev, device, len(token_ids) if token_ids else 0,
                    )
                except Exception as _dbg_e:
                    _logger.warning("[TE Patch 调试] 获取调试信息失败: %s", _dbg_e)

                # L2 模式：TE 在 CPU 时，临时上 GPU 编码（快速），编码后立即下 CPU
                # v0.6.18+ 修复：不检查 device 参数是否为 cuda，因为 diffusers 的
                # components._execution_device 会根据 TE 设备推导，TE 在 CPU 时返回 cpu。
                # 只需检查 TE 在 CPU 且引擎配置设备是 cuda 即可。
                #
                # v0.6.22+ 修复：Ref2V/VideoEdit 模式 prompt 含大量 vision tokens
                # （单图 ~7748, 双图 ~15449）。
                # v0.6.23+ 修订：实测 CPU 编码 7748 token 卡死 17min+（attention O(n²) + SDNQ
                # 4bit 量化 torch.compile 慢），反而 GPU 编码几十秒能完成。
                # 取消强制 CPU 编码阈值，让长 token 走 GPU 编码（L2 模式下 TE 临时上 GPU
                # 编码后立即下 CPU，与短 token 同路径，避免 CPU 卡死）。
                # 仅当 _device 不是 cuda 或 offload 不是 L2 时才走 CPU 编码（由上面分支处理）。
                _token_count = len(token_ids) if token_ids is not None else 0
                # v0.7.7+ T2.2 conditioning 磁盘缓存：命中则直接返回，跳过 TE 编码
                # （含 L2 临时上 GPU 的 41GB 峰值）。key 由 wrapper.generate() 计算，
                # 仅单次生成有效（推理完成后 finally 重置）。
                _cond_key = getattr(self, "_cond_cache_key", None)
                _cond_cache = getattr(self, "_cond_cache", None)
                if _cond_key and _cond_cache is not None:
                    _cached_embeds = _cond_cache.get(_cond_key)
                    # v0.8.x 修复：不同分辨率/帧数下同一 prompt+参考图会得到不同
                    # conditioning token 数，但旧 cache key（variant+mode+prompt+文件指纹）
                    # 未纳入尺寸 → 不同分辨率任务误命中同一缓存条目，text_indices(S)
                    # 与 text_embeds(≠S) 不匹配 → index_copy_() 崩（269 vs 465 发源地）。
                    # 命中但 token 数不一致 → 一律视为未命中，重新编码（尺寸不符是误命中）。
                    if (_cached_embeds is not None and _token_count
                            and _cached_embeds.shape[1] != _token_count):
                        _logger.warning(
                            "[cond-cache] 命中条目 token=%d 与当前请求 token=%d 不符（尺寸差异，"
                            "视为误命中，重新编码）",
                            _cached_embeds.shape[1], _token_count,
                        )
                        _cached_embeds = None
                    if _cached_embeds is not None:
                        _ret_device = device
                        if _ret_device is None:
                            try:
                                _ret_device = next(text_encoder.parameters()).device
                            except Exception:
                                _ret_device = None
                        if _ret_device is not None or dtype is not None:
                            _cached_embeds = _cached_embeds.to(device=_ret_device, dtype=dtype)
                        _logger.info(
                            "[cond-cache] conditioning 命中缓存，跳过 TE 编码（token=%d）",
                            _token_count,
                        )
                        return _cached_embeds
                # v0.6.23+ 取消强制 CPU，所有 token 走 GPU 编码
                _force_cpu_encode = False
                # v0.7.7+ T2.1 动态显存预检：L2 模式下 TE 上 GPU 前检查空闲显存，
                # 若"卸载 vae/transformer 可腾空间后"仍不足编码 headroom，则本次
                # 降级 CPU 编码（等同 L1 行为），避免 TE 上 GPU 触发 sysmem fallback
                # 颠簸（GPU 100% 实为 PCIe 页迁移，编码 10min+ 不返回）。
                if not _force_cpu_encode and _offload_level == "L2" and _device.startswith("cuda"):
                    try:
                        import torch as _t
                        if _t.cuda.is_available():
                            _free_b, _total_b = _t.cuda.mem_get_info()
                            _total_mb = _total_b // (1024 * 1024)
                            _used_mb = _total_mb - _free_b // (1024 * 1024)
                            _te_size_mb = sum(
                                p.numel() * p.element_size() for p in text_encoder.parameters()
                            ) // (1024 * 1024)
                            _needed_mb = 4096 + int(_token_count * 2.0)
                            _offloadable_mb = 0
                            _p_ref = getattr(text_encoder, "_aireliable_pipe_ref", None)
                            if _p_ref is not None:
                                for _cn in ("vae", "audio_vae", "transformer", "transformer_ref"):
                                    _c = getattr(_p_ref, _cn, None)
                                    if (_c is not None and hasattr(_c, "device")
                                            and str(_c.device).startswith("cuda")):
                                        _offloadable_mb += sum(
                                            pp.numel() * pp.element_size()
                                            for pp in _c.parameters()
                                        ) // (1024 * 1024)
                            _free_after_all_mb = _total_mb - (_used_mb + _te_size_mb) + _offloadable_mb
                            if _free_after_all_mb < _needed_mb:
                                _force_cpu_encode = True
                                _logger.warning(
                                    "[T2.1 显存预检] L2 TE 上 GPU 空间不足（即使卸载 vae/transformer）: "
                                    "编码后空余≈%dMB < 需 %dMB（token=%d, TE≈%dMB, 可腾≈%dMB, 总显存=%dMB），"
                                    "本次降级 CPU 编码（等同 L1）",
                                    _free_after_all_mb, _needed_mb, _token_count,
                                    _te_size_mb, _offloadable_mb, _total_mb,
                                )
                            else:
                                _logger.info(
                                    "[T2.1 显存预检] L2 TE 可上 GPU: 卸载可腾≈%dMB, 编码后空余≈%dMB >= 需 %dMB",
                                    _offloadable_mb, _free_after_all_mb, _needed_mb,
                                )
                    except Exception as _e:
                        _logger.warning("[T2.1 显存预检] 异常（保持默认行为）: %s", _e)
                _logger.info(
                    "[TE Patch] token_ids_len=%s, force_cpu=%s, offload=%s, device=%s",
                    _token_count, _force_cpu_encode, _offload_level, _device,
                )

                # v0.7.5 修复：TE attention 强制 eager → sdpa（长 token 编码显存爆炸根因）
                # 根因：sdnq.load_sdnq_model 不经过 transformers.from_pretrained 的
                # sdpa 自动选择，config._attn_implementation 保持默认 "eager"。
                # eager 在 64 heads × 7350 token 下材料化完整 attention 矩阵：
                # (1, 64, 7350, 7350) bf16 ≈ 6.9GB + fp32 softmax 副本 ≈ 13.8GB，
                # 叠加 TE 权重 20GB 后挤爆 48GB 显存 → Windows sysmem fallback
                # 颠簸（GPU "100%" 实为 PCIe 页迁移，编码 10min+ 不返回；
                # 历史日志中 4164/7350/7748 token 编码全部失败，仅纯文本通过）。
                # sdpa 内存线性（单层峰值 <1GB），CPU 编码（L1/L3）同样受益。
                # 注意：config._attn_implementation 在每次 forward 时解析
                # （ALL_ATTENTION_FUNCTIONS.get_interface），运行时切换即时生效。
                try:
                    _patched_cfgs = []
                    _seen_cfg_ids = set()
                    _diag_impls = []  # v0.7.5 诊断：无条件记录所有 config 的实现
                    for _mod in text_encoder.modules():
                        _cfg = getattr(_mod, "config", None)
                        if _cfg is None or id(_cfg) in _seen_cfg_ids:
                            continue
                        _seen_cfg_ids.add(id(_cfg))
                        _impl = getattr(_cfg, "_attn_implementation", None)
                        _diag_impls.append(f"{type(_cfg).__name__}={_impl!r}")
                        # v0.7.5 修订：transformers 5.x 的 config._attn_implementation
                        # 默认为 None（而非 "eager"），sdnq 加载不经过 from_pretrained
                        # 的解析流程 → None 保留 → get_interface(None, eager) 回退
                        # eager attention。因此 None 和 "eager" 都需要切到 sdpa；
                        # 显式 flash/flash_attention_2 则保留不动。
                        if _impl is None or _impl == "eager":
                            _cfg._attn_implementation = "sdpa"
                            _patched_cfgs.append(
                                f"{type(_cfg).__name__}({('None' if _impl is None else _impl)}→sdpa)")
                    _logger.info(
                        "[TE attention 诊断] config 实现明细: [%s]%s",
                        "; ".join(_diag_impls),
                        f"，已切换: {', '.join(_patched_cfgs)}" if _patched_cfgs else "（无需切换）",
                    )
                except Exception as _attn_e:
                    _logger.warning("[TE attention] sdpa 切换失败（保持原实现）: %s", _attn_e)
                # v0.7.6 方案1-b：确定性 forward 级强制 SDPA。
                # 上面的动态 config 循环只改 config 值，无法 100% 保证命中运行时 attention
                # 读到的同一 config 对象（v0.7.5 实测 ref2va 长 token 仍停滞）。
                # 此处直接替换每个文本 attention 模块的 forward：调用原始 forward 前把
                # 该模块自身 config._attn_implementation 置为 "sdpa"，确保 transformers 5.x
                # Qwen3VLTextAttention.forward 内的 get_interface 必然解析到 sdpa（线性显存），
                # 同时消除 O(n²) EAGER 矩阵材料化 → 不再击穿 48GB 触发 sysmem 颠簸。
                try:
                    _attn_sdpa_patched = 0
                    for _mod in text_encoder.modules():
                        _attn_cls = type(_mod)
                        _attn_name = getattr(_attn_cls, "__name__", "") or ""
                        # 仅处理文本注意力（Qwen3VLTextAttention/Qwen3Attention）；跳过 vision
                        # attention（VisionAttention 本身已按 sdpa 分发，无需覆盖）。
                        if "TextAttention" not in _attn_name:
                            continue
                        if getattr(_mod, "_aireliable_sdpa_fwd", False):
                            continue
                        _orig_attn_fwd = _mod.forward

                        def _sdpa_forced_forward(*_a, _m=_mod, _orig=_orig_attn_fwd, **_kw):
                            _cfg = getattr(_m, "config", None)
                            if _cfg is not None and getattr(
                                    _cfg, "_attn_implementation", None) != "sdpa":
                                _cfg._attn_implementation = "sdpa"
                            return _orig(*_a, **_kw)

                        _mod.forward = _sdpa_forced_forward
                        _mod._aireliable_sdpa_fwd = True
                        _attn_sdpa_patched += 1
                    _logger.info(
                        "[TE attention] forward 级强制 sdpa 模块数=%d（%s）",
                        _attn_sdpa_patched,
                        "ok" if _attn_sdpa_patched else "未命中文本 attention 模块",
                    )
                except Exception as _attn_fwd_e:
                    _logger.warning("[TE attention] forward 级 sdpa 覆盖失败: %s", _attn_fwd_e)
                if (_offload_level == "L2" and _device.startswith("cuda")
                        and not _force_cpu_encode):
                    try:
                        te_device = next(text_encoder.parameters()).device
                        if te_device.type == "cpu":
                            # 1. 先卸载 vae/audio_vae 到 CPU（腾显存给 TE）
                            # 通过 pipe 引用获取 vae 组件
                            _pipe = getattr(text_encoder, "_aireliable_pipe_ref", None)
                            if _pipe is not None:
                                for vae_name in ("vae", "audio_vae"):
                                    try:
                                        vae_comp = getattr(_pipe, vae_name, None)
                                        if (vae_comp is not None and hasattr(vae_comp, "device")
                                                and str(vae_comp.device).startswith("cuda")):
                                            vae_comp.to("cpu")
                                            _vae_offloaded.append(vae_name)
                                    except Exception:
                                        pass
                                if _vae_offloaded:
                                    torch.cuda.empty_cache()
                                    _logger.info(
                                        "[L2 VAE 临时下 CPU] %s（腾显存给 text_encoder）",
                                        ", ".join(_vae_offloaded),
                                    )

                            # 1.5 v0.7.3+ 修复：长 token 编码（Ref2V/I2V 等图片模式，
                            # 单图 ~7350 token）时 TE 上 GPU 后剩余显存不足（实测
                            # 48489/49152MB 仅剩 663MB），KV cache/激活分配触发
                            # Windows CUDA sysmem fallback 显存颠簸——GPU "100%"
                            # 实为 PCIe 页迁移，编码 45min+ 不返回。
                            # 对策：估算编码所需 headroom，不足时把 transformer
                            # 也临时下 CPU（编码完成后对称恢复）。
                            try:
                                _total_mb = torch.cuda.get_device_properties(
                                    _device).total_memory // (1024 * 1024)
                            except Exception:
                                _total_mb = 49152
                            _te_size_mb = sum(
                                p.numel() * p.element_size()
                                for p in text_encoder.parameters()
                            ) // (1024 * 1024)
                            # 编码期额外显存需求估算（v0.7.4 修订）：
                            # v0.7.3 的 0.6MB/token 只算了 KV cache（线性项），
                            # 严重低估 O(n²) attention 激活（SDNQ 4bit + torch.compile
                            # 下未启用 flash attention，heads × n² 激活远超 KV cache）。
                            # 实测 7350 token：KV 估算 4.4GB 实际触发 sysmem fallback
                            # 颠簸 10min+ 不返回。改为 2MB/token + 4GB 基础的保守估算：
                            # 7350 token → ~18.8GB，强制腾出 transformer 空间；
                            # 短 prompt（<2k token）→ <8GB，不触发，行为与 v0.7.3 一致。
                            _needed_free_mb = 4096 + int(_token_count * 2.0)
                            _used_mb = _get_gpu_mem_mb() or 0
                            _free_after_te_mb = _total_mb - (_used_mb + _te_size_mb)
                            if _free_after_te_mb < _needed_free_mb:
                                _logger.info(
                                    "[L2 编码 headroom 检查] 剩余显存不足: token=%d, "
                                    "TE≈%dMB, TE上GPU后空余≈%dMB, 编码需空余≈%dMB, 总显存=%dMB",
                                    _token_count, _te_size_mb, _free_after_te_mb,
                                    _needed_free_mb, _total_mb,
                                )
                            if (_free_after_te_mb < _needed_free_mb
                                    and _pipe is not None):
                                for _tr_name in ("transformer", "transformer_ref"):
                                    try:
                                        _tr_comp = getattr(_pipe, _tr_name, None)
                                        if (_tr_comp is not None
                                                and hasattr(_tr_comp, "device")
                                                and str(_tr_comp.device).startswith("cuda")):
                                            _tr_comp.to("cpu")
                                            torch.cuda.empty_cache()
                                            _transformer_offloaded.append(_tr_name)
                                    except Exception:
                                        pass
                                if _transformer_offloaded:
                                    _logger.info(
                                        "[L2 transformer 临时下 CPU] %s（长 token 编码需腾显存: "
                                        "token=%d, TE≈%dMB, 编码需空余≈%dMB, 总显存=%dMB）",
                                        ", ".join(_transformer_offloaded), _token_count,
                                        _te_size_mb, _needed_free_mb, _total_mb,
                                    )

                            # 2. TE 上 GPU
                            _vram_before = _get_gpu_mem_mb()
                            text_encoder.to(_device)
                            torch.cuda.empty_cache()
                            _vram_after = _get_gpu_mem_mb()
                            _te_moved_to_gpu = True
                            _logger.info(
                                "[L2 TE 临时上 GPU] text_encoder 已移到 GPU（显存: %sMB → %sMB）",
                                _vram_before, _vram_after,
                            )
                    except Exception as _e:
                        _logger.warning("[L2 TE 临时上 GPU] 失败，回退到 CPU 编码: %s", _e)
                        # 失败时恢复 vae 到 GPU
                        _pipe = getattr(text_encoder, "_aireliable_pipe_ref", None)
                        if _pipe is not None:
                            for vae_name in _vae_offloaded:
                                try:
                                    vae_comp = getattr(_pipe, vae_name, None)
                                    if vae_comp is not None:
                                        vae_comp.to(_device)
                                except Exception:
                                    pass
                        _vae_offloaded = []
                        _te_moved_to_gpu = False

                # L1/L3 模式：TE 在 CPU 时，强制 device=cpu 避免设备冲突
                if not _te_moved_to_gpu:
                    try:
                        te_device = next(text_encoder.parameters()).device
                        if (te_device.type == "cpu" and device is not None
                                and str(device).startswith("cuda")):
                            device = te_device
                    except Exception:
                        pass

                # 调用原始编码函数
                # v0.6.18+ 修复：TE 上 GPU 后，device 参数也要改为 GPU，否则原始函数
                # 内部创建的 input_ids 等张量在 CPU，与 GPU TE 设备冲突
                if _te_moved_to_gpu:
                    import torch as _torch
                    device = _torch.device(_device)
                result = _orig_get_embeds(text_encoder, processor, token_ids, vision_inputs,
                                          text_encoder_layer, device, dtype)

                # v0.7.7+ T2.2：编码完成写入 conditioning 缓存（仅当本次请求已计算 key）
                if result is not None and _cond_key and _cond_cache is not None:
                    try:
                        _cond_cache.put(_cond_key, result)
                    except Exception as _ce:
                        _logger.warning("[cond-cache] 写入失败: %s", _ce)

                # L2 模式：编码完成后立即把 TE 下 CPU，恢复 VAE 到 GPU
                if _te_moved_to_gpu:
                    try:
                        _vram_before = _get_gpu_mem_mb()
                        text_encoder.to("cpu")
                        torch.cuda.empty_cache()
                        _vram_after = _get_gpu_mem_mb()
                        _freed = (_vram_before - _vram_after
                                  if _vram_before and _vram_after else 0)
                        _logger.info(
                            "[L2 TE 编码完成] text_encoder 已移回 CPU（释放 %sMB 显存）",
                            _freed,
                        )
                        # v0.7.3+ 恢复临时下 CPU 的 transformer（先于 VAE，推理主干必须回 GPU）
                        _pipe = getattr(text_encoder, "_aireliable_pipe_ref", None)
                        if _pipe is not None:
                            for _tr_name in _transformer_offloaded:
                                try:
                                    _tr_comp = getattr(_pipe, _tr_name, None)
                                    if _tr_comp is not None:
                                        _tr_comp.to(_device)
                                        _logger.info("[L2 transformer 恢复] %s 已移回 GPU", _tr_name)
                                except Exception as _e:
                                    _logger.warning("[L2 transformer 恢复] %s 失败: %s", _tr_name, _e)
                            torch.cuda.empty_cache()
                        _transformer_offloaded = []
                        # 恢复 vae/audio_vae 到 GPU
                        _pipe = getattr(text_encoder, "_aireliable_pipe_ref", None)
                        if _pipe is not None:
                            for vae_name in _vae_offloaded:
                                try:
                                    vae_comp = getattr(_pipe, vae_name, None)
                                    if vae_comp is not None:
                                        vae_comp.to(_device)
                                        _logger.info("[L2 VAE 恢复] %s 已移回 GPU", vae_name)
                                except Exception as _e:
                                    _logger.warning("[L2 VAE 恢复] %s 失败: %s", vae_name, _e)
                            torch.cuda.empty_cache()
                    except Exception as _e:
                        _logger.warning("[L2 TE 编码完成] 移回 CPU 失败: %s", _e)

                return result

            _patched_get_embeds._aireliable_patched = True
            encoders_module.get_qwen3vl_prompt_embeds = _patched_get_embeds

            self._logger.info(
                "get_qwen3vl_prompt_embeds 已 patch（L2 模式: TE 临时上 GPU 编码后下 CPU; L1/L3: CPU 编码）"
            )
        except Exception as e:
            self._logger.warning("get_qwen3vl_prompt_embeds patch 失败: %s", e)

    # ============ CPU offload 策略 ============

    def _resolve_sdnq_load_device(self, subdir: str) -> Any:
        """v0.6.11+ 根据 offload_level 和组件类型决定 SDNQ 加载设备。

        利用 CPU Offload 等级，显存不够时用内存（RAM）。
        text_encoder（19.6GB）在 prompt 编码后不再需要，可以加载到 CPU；
        transformer（17.5GB）是推理必需，需要加载到 GPU（L0/L1/L2）或按需移动（L3）。

        Args:
            subdir: 组件子目录名（text_encoder / transformer / transformer_ref）

        Returns:
            torch.device: 加载目标设备
        """
        import torch

        # 非 CUDA 设备：全部加载到 CPU
        if not self._device.startswith("cuda"):
            return torch.device("cpu")

        is_text_encoder = subdir == "text_encoder"
        is_transformer = subdir in ("transformer", "transformer_ref")

        if self._offload_level == "L0":
            # L0：全部加载到 GPU（需要 48GB+ 显存）
            return torch.device(self._device)
        elif self._offload_level in ("L1", "L2"):
            # L1/L2：text_encoder 加载到 CPU（用 RAM），transformer 加载到 GPU
            # 这样 GPU 只占 transformer (17.5GB) + vae (5.8GB) ≈ 23GB，剩余 25GB 用于 attention
            if is_text_encoder:
                self._logger.info(
                    "[CPU Offload L%s] text_encoder 加载到 CPU（释放 ~19.6GB 显存给 attention）",
                    self._offload_level,
                )
                return torch.device("cpu")
            else:
                return torch.device(self._device)
        elif self._offload_level == "L3":
            # L3：全部加载到 CPU（用 RAM），推理时按需把 transformer 移到 GPU
            # 适用于显存严重不足（< 24GB）的场景
            self._logger.info(
                "[L3] %s 加载到 CPU（推理时按需移到 GPU）",
                subdir,
            )
            return torch.device("cpu")
        else:
            # 未知级别，按 L1 处理
            if is_text_encoder:
                return torch.device("cpu")
            return torch.device(self._device)

    def _fix_rope_device(self) -> None:
        """v0.6.11+ 修复 SDNQ 量化后 transformer.rope.inv_freq buffer 设备不一致问题。

        SDNQ 4bit 量化的 transformer 在 .to(device) 时，rope 模块的 inv_freq buffer
        可能没有被正确移动到 GPU，导致推理时 position_ids(cuda) 与 inv_freq(cpu)
        设备冲突，抛出 "Expected all tensors to be on the same device" 错误。

        本方法在 offload 策略应用后，显式检查并修复 rope 的所有 buffer 设备。
        """
        import torch

        try:
            # 查找 transformer 组件（FL2VA 用 transformer，Ref2VA 用 transformer_ref）
            transformer = None
            for attr_name in ("transformer", "transformer_ref"):
                tr = getattr(self._pipe, attr_name, None)
                if tr is not None:
                    transformer = tr
                    break

            if transformer is None:
                return

            if not hasattr(transformer, "rope"):
                return

            rope = transformer.rope
            try:
                target_device = next(transformer.parameters()).device
            except StopIteration:
                return

            fixed_count = 0
            # 修复 inv_freq（最关键的 buffer）
            if hasattr(rope, "inv_freq"):
                if rope.inv_freq.device != target_device:
                    rope.inv_freq.data = rope.inv_freq.data.to(target_device)
                    fixed_count += 1
                    self._logger.info(
                        "[RoPE Fix] inv_freq 已移到 %s（之前在 %s）",
                        target_device, rope.inv_freq.device,
                    )

            # 修复 rope 的所有 buffer
            for buf_name, buf in rope.named_buffers():
                if buf.device != target_device:
                    buf.data = buf.data.to(target_device)
                    fixed_count += 1
                    self._logger.info(
                        "[RoPE Fix] buffer %s 已移到 %s",
                        buf_name, target_device,
                    )

            if fixed_count > 0:
                self._logger.info("[RoPE Fix] 共修复 %d 个 buffer 设备", fixed_count)
            else:
                self._logger.debug("[RoPE Fix] 所有 buffer 设备正确，无需修复")

        except Exception as e:
            self._logger.warning("[RoPE Fix] rope 设备修复失败: %s", e)

    def _patch_decoders_device(self) -> None:
        """v0.6.21+ 重写 VAE 解码步骤，使用 VAE 自身设备而非 _execution_device。

        问题：
            L2 策略下 components._execution_device 返回 CPU（因为 text_encoder 在 CPU，
            _execution_device 遍历 components 时返回第一个模型即 text_encoder 的设备）。
            但 VAE 在 GPU（已通过 [L2 VAE 恢复] 移回 GPU），latents 也在 GPU（来自扩散步骤）。
            原 __call__ 用 _execution_device 创建 latents_std/latents_mean（在 CPU），
            与 block_state.latents（在 GPU）相乘时报 "Expected all tensors to be on the same device"。

        方案：
            完全替换 MiniMaxH3VideoDecodeStep / MiniMaxH3AudioDecodeStep 的 __call__，
            用 VAE 自身的设备（next(vae.parameters()).device）替代 _execution_device。
            同时显式将 latents 移到 VAE 设备上，确保设备一致。

        设计原则：
            - 只在 L1/L2/L3 offload 模式下 patch（L0 不需要，所有 tensor 都在 GPU）
            - patch 失败时降级到原始 __call__（try/except 包裹）
            - 保留原始 @torch.no_grad() 语义（用 with torch.no_grad() 包裹）
        """
        # L0 模式不需要 patch（所有 tensor 都在 GPU）
        if self._offload_level == "L0":
            return

        try:
            from diffusers.modular_pipelines.minimax_h3.decoders import (
                MiniMaxH3VideoDecodeStep,
                MiniMaxH3AudioDecodeStep,
            )
        except ImportError:
            self._logger.warning("无法导入 decoders 模块，跳过 decoders patch")
            return

        import torch
        _logger = self._logger

        # ---- 替换 MiniMaxH3VideoDecodeStep.__call__ ----
        if not getattr(MiniMaxH3VideoDecodeStep, "_aireliable_decoders_patched", False):
            _orig_video_decode = MiniMaxH3VideoDecodeStep.__call__

            def _patched_video_decode(block_self, components, state):
                """用 VAE 设备替代 _execution_device 的 VideoDecode。"""
                try:
                    with torch.no_grad():
                        block_state = block_self.get_block_state(state)
                        # 关键修复：使用 VAE 自身设备（GPU），而非 _execution_device（可能返回 CPU）
                        device = next(components.vae.parameters()).device

                        if block_state.output_type not in ("pil", "np", "pt"):
                            raise ValueError(
                                f"`output_type` must be one of 'pil', 'np' or 'pt', got {block_state.output_type!r}."
                            )

                        latents_mean = torch.tensor(
                            components.vae.config.latents_mean, device=device
                        ).view(1, -1, 1, 1, 1)
                        latents_std = torch.tensor(
                            components.vae.config.latents_std, device=device
                        ).view(1, -1, 1, 1, 1)

                        # 显式确保 latents 在 VAE 设备上
                        if hasattr(block_state.latents, "device") and block_state.latents.device != device:
                            _logger.info(
                                "[Decoder Device Fix] VideoDecode.latents %s -> %s",
                                block_state.latents.device, device,
                            )
                            block_state.latents = block_state.latents.to(device)

                        latents = block_state.latents * latents_std + latents_mean
                        
                        # 检查并修复 NaN 的 latent。
                        # v1.2+（41 号任务）：占比超过阈值视为数值发散（扩散已产出垃圾 latent），
                        # 直接抛 RuntimeError 让任务落 failed，绝不替换为 0 输出"灰屏假成功"；
                        # 仅占比微小（< 阈值）时才 nan_to_num 兜底，避免个位噪声点导致全黑视频。
                        _numel = latents.numel() or 1
                        n_nan = torch.isnan(latents).sum().item()
                        n_inf = torch.isinf(latents).sum().item()
                        n_bad = n_nan + n_inf
                        _frac = n_bad / _numel
                        if n_bad > 0:
                            if _frac > _NAN_FAIL_FRACTION:
                                _logger.error(
                                    "[MiniMax-H3 NaN 检测] video latents 数值发散，终止任务："
                                    "%d NaN + %d Inf = %d 个 / 共 %d，占比 %.4f%%（> %.4f%%，非偶发噪声）",
                                    n_nan, n_inf, n_bad, _numel, _frac * 100,
                                    _NAN_FAIL_FRACTION * 100,
                                )
                                raise RuntimeError(
                                    "视频 latent 数值发散（NaN/Inf 占比 %.4f%%），扩散未收敛，"
                                    "已终止以避免输出灰屏视频。请调低分辨率长宽比、使用横屏默认尺寸 "
                                    "或降低 CFG 后重试。" % (_frac * 100)
                                )
                            _logger.warning(
                                "[MiniMax-H3 NaN 检测] 发现 %d NaN + %d Inf（占比 %.4f%%<阈值），已替换为 0",
                                n_nan, n_inf, _frac * 100,
                            )
                            latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

                        with torch.autocast(
                            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                        ):
                            video = components.vae.decode(latents, return_dict=False)[0]
                        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
                        pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
                        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
                        block_state.videos = components.video_processor.postprocess_video(
                            video, output_type=block_state.output_type
                        )

                        block_self.set_block_state(state, block_state)
                        return components, state
                except RuntimeError as _e:
                    # 数值发散（NaN/Inf）导致的失败必须向上传播，绝不降级到原始解码再吞错
                    if "数值发散" in str(_e):
                        raise
                    _logger.warning(
                        "[Decoder Device Fix] VideoDecode 重写失败（降级到原始）: %s: %s",
                        type(_e).__name__, _e,
                    )
                    return _orig_video_decode(block_self, components, state)
                except Exception as _e:
                    _logger.warning(
                        "[Decoder Device Fix] VideoDecode 重写失败（降级到原始）: %s: %s",
                        type(_e).__name__, _e,
                    )
                    return _orig_video_decode(block_self, components, state)

            MiniMaxH3VideoDecodeStep.__call__ = _patched_video_decode
            MiniMaxH3VideoDecodeStep._aireliable_decoders_patched = True
            self._logger.info("MiniMaxH3VideoDecodeStep.__call__ 已重写（使用 VAE 设备替代 _execution_device）")

        # ---- 替换 MiniMaxH3AudioDecodeStep.__call__ ----
        if not getattr(MiniMaxH3AudioDecodeStep, "_aireliable_decoders_patched", False):
            _orig_audio_decode = MiniMaxH3AudioDecodeStep.__call__

            def _patched_audio_decode(block_self, components, state):
                """用 audio_vae 设备替代 _execution_device 的 AudioDecode。"""
                try:
                    with torch.no_grad():
                        block_state = block_self.get_block_state(state)
                        # 关键修复：使用 audio_vae 自身设备（GPU）
                        device = next(components.audio_vae.parameters()).device

                        audio_latents_mean = torch.tensor(
                            components.audio_vae.config.latents_mean, device=device
                        ).view(1, -1, 1)
                        audio_latents_std = torch.tensor(
                            components.audio_vae.config.latents_std, device=device
                        ).view(1, -1, 1)

                        # 显式确保 audio_latents 在 audio_vae 设备上
                        if hasattr(block_state.audio_latents, "device") and block_state.audio_latents.device != device:
                            _logger.info(
                                "[Decoder Device Fix] AudioDecode.audio_latents %s -> %s",
                                block_state.audio_latents.device, device,
                            )
                            block_state.audio_latents = block_state.audio_latents.to(device)

                        audio_latents = block_state.audio_latents * audio_latents_std + audio_latents_mean

                        # v1.2+（41 号任务）audio latent 数值发散检查：
                        # 与 video 分支同阈值，区分"音频分支先发散"这一根因方向。
                        _anumel = audio_latents.numel() or 1
                        _a_nan = torch.isnan(audio_latents).sum().item()
                        _a_inf = torch.isinf(audio_latents).sum().item()
                        _a_bad = _a_nan + _a_inf
                        _a_frac = _a_bad / _anumel
                        if _a_bad > 0:
                            if _a_frac > _NAN_FAIL_FRACTION:
                                _logger.error(
                                    "[MiniMax-H3 NaN 检测] audio latents 数值发散（音频分支先行发散），终止任务："
                                    "%d NaN + %d Inf / 共 %d，占比 %.4f%%",
                                    _a_nan, _a_inf, _anumel, _a_frac * 100,
                                )
                                raise RuntimeError(
                                    "音频 latent 数值发散（NaN/Inf 占比 %.4f%%），扩散未收敛，已终止。"
                                    "请调低分辨率长宽比或使用横屏默认尺寸后重试。" % (_a_frac * 100)
                                )
                            _logger.warning(
                                "[MiniMax-H3 NaN 检测] audio latents 微扰（占比 %.4f%%<阈值），已替换为 0",
                                _a_frac * 100,
                            )
                            audio_latents = torch.nan_to_num(
                                audio_latents, nan=0.0, posinf=0.0, neginf=0.0
                            )

                        audio = components.audio_vae.decode(audio_latents, return_dict=False)[0]
                        block_state.audio = audio.float().permute(1, 0, 2)
                        block_state.sampling_rate = components.audio_sampling_rate

                        block_self.set_block_state(state, block_state)
                        return components, state
                except RuntimeError as _e:
                    # 数值发散（NaN/Inf）导致的失败必须向上传播
                    if "数值发散" in str(_e):
                        raise
                    _logger.warning(
                        "[Decoder Device Fix] AudioDecode 重写失败（降级到原始）: %s: %s",
                        type(_e).__name__, _e,
                    )
                    return _orig_audio_decode(block_self, components, state)
                except Exception as _e:
                    _logger.warning(
                        "[Decoder Device Fix] AudioDecode 重写失败（降级到原始）: %s: %s",
                        type(_e).__name__, _e,
                    )
                    return _orig_audio_decode(block_self, components, state)

            MiniMaxH3AudioDecodeStep.__call__ = _patched_audio_decode
            MiniMaxH3AudioDecodeStep._aireliable_decoders_patched = True
            self._logger.info("MiniMaxH3AudioDecodeStep.__call__ 已重写（使用 audio_vae 设备替代 _execution_device）")

    def _patch_encoders_device(self) -> None:
        """Patch VAE encoder 让 encode_vae_condition 在 CPU 上执行。

        问题：
            encoders.py:133 `vae.encode(pixels, ...)` 触发 `aten::slow_conv3d_forward`，
            该算子只注册在 CPU 后端，CUDA 后端无实现（cuDNN 缺该路径）。
            T2V 不调用 encoder 所以正常；I2V/FL2V/Ref2V/video_edit 都会失败。

        方案：
            Patch `diffusers.modular_pipelines.minimax_h3.encoders.encode_vae_condition`，
            临时把 vae 和 pixels 移到 CPU 做 encode，编码后再把 vae 移回原设备。
            encode_vae_condition 本身已经把结果 .cpu() 返回（line 133），所以无侵入。

        设计原则：
            - 模块级函数 patch（只 patch 一次，幂等）
            - 保留 vae 原设备（避免影响后续 vae.decode）
            - 异常时回退到原始函数（让原始错误抛出便于排查）
        """
        try:
            from diffusers.modular_pipelines.minimax_h3 import encoders as mm_encoders
        except ImportError:
            self._logger.warning("无法导入 minimax_h3.encoders 模块，跳过 encoders patch")
            return

        if getattr(mm_encoders, "_aireliable_encoders_patched", False):
            return  # 已 patch

        import torch
        _orig_encode_vae_condition = mm_encoders.encode_vae_condition
        _logger = self._logger

        def _patched_encode_vae_condition(vae, pixels, pixel_mean, pixel_std, encode_seed=42):
            """强制 vae.encode 在 CPU 上跑，避免 slow_conv3d_forward CUDA 缺失。"""
            try:
                orig_device = next(vae.parameters()).device
                if orig_device.type == "cpu":
                    # 已经在 CPU，直接调用原函数
                    return _orig_encode_vae_condition(
                        vae, pixels, pixel_mean, pixel_std, encode_seed
                    )
                # 临时把 vae 移到 CPU
                vae.to("cpu")
                pixels = pixels.to("cpu")
                _logger.info(
                    "[Encoder CPU Patch] vae.encode %s -> cpu（avoid slow_conv3d CUDA）",
                    orig_device,
                )
                try:
                    return _orig_encode_vae_condition(
                        vae, pixels, pixel_mean, pixel_std, encode_seed
                    )
                finally:
                    # 恢复 vae 到原设备
                    vae.to(orig_device)
            except Exception as _e:
                _logger.warning(
                    "[Encoder CPU Patch] encode_vae_condition 重写失败，降级到原始: %s: %s",
                    type(_e).__name__, _e,
                )
                # 降级到原始函数（让原始错误抛出便于排查）
                return _orig_encode_vae_condition(
                    vae, pixels, pixel_mean, pixel_std, encode_seed
                )

        mm_encoders.encode_vae_condition = _patched_encode_vae_condition
        mm_encoders._aireliable_encoders_patched = True
        self._logger.info(
            "encode_vae_condition 已 patch（VAE encode 强制 CPU，规避 slow_conv3d CUDA 缺失）"
        )

        # v0.7.8+ 修复：ref2va 音频条件编码块（encoders.py:749）直接调用
        #   audio_vae.encode(reference.audio.to(components._execution_device))
        #   L2 offload 下 _execution_device=CPU，但 audio_vae 权重在 GPU
        #   → "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)"
        #     device mismatch（audio 与 video 不同，走 2D conv 无 slow_conv3d 缺失，
        #     可安全在 audio_vae 自身设备上编码）。
        # 方案：patch audio_vae.encode，把 CPU 输入移动到 audio_vae 当前设备再编码。
        try:
            _audio_vae = getattr(self._pipe, "audio_vae", None)
            import torch as _torch
            if (_audio_vae is not None and hasattr(_audio_vae, "encode")
                    and not getattr(_audio_vae, "_aireliable_audio_encode_patched", False)):
                _orig_audio_encode = _audio_vae.encode

                # 注意：普通函数赋给实例属性不会绑定 self，所以签名与音频条件编码
                # 的调用方一致（encoders.py:749 `audio_vae.encode(sample, return_dict=False)`）。
                def _patched_audio_encode(sample, *_a, **_kw):
                    try:
                        _dev = next(_audio_vae.parameters()).device
                        if _dev.type == "cuda" and isinstance(sample, _torch.Tensor) and sample.device != _dev:
                            sample = sample.to(_dev)
                            self._logger.debug(
                                "[Audio Encode Device Fix] input %s -> %s", sample.device, _dev)
                        return _orig_audio_encode(sample, *_a, **_kw)
                    except Exception as _e:
                        self._logger.warning(
                            "[Audio Encode Device Fix] audio_vae.encode 重写失败，降级到原始: %s: %s",
                            type(_e).__name__, _e,
                        )
                        return _orig_audio_encode(sample, *_a, **_kw)

                _audio_vae.encode = _patched_audio_encode
                _audio_vae._aireliable_audio_encode_patched = True
                self._logger.info(
                    "audio_vae.encode 已 patch（输入移动到 audio_vae 设备，修复 ref2va 音频条件编码 device mismatch）"
                )
        except Exception as _e:
            self._logger.warning("audio_vae.encode patch 失败: %s", _e)

    def _apply_offload_strategy(self, level: str) -> None:
        """根据 offload 级别应用加载策略。

        Args:
            level: offload 级别（L0 / L1 / L2 / L3）
        """
        if level == "L0":
            self._load_all_to_gpu()
        elif level == "L1":
            self._load_with_te_offload()
        elif level == "L2":
            self._load_with_layer_offload()
        elif level == "L3":
            self._load_with_aggressive_offload()
        else:
            self._logger.warning("未知 offload 级别，按 L1 处理: %s", level)
            self._load_with_te_offload()

        # v0.6.11+ 统一修复：所有 offload 级别都检查并修复 rope.inv_freq 设备
        # SDNQ 量化后 .to(device) 可能跳过非量化 buffer，导致推理时设备冲突
        self._fix_rope_device()

        # v0.6.20+ 修复：VAE 解码前 latents 可能在 CPU（scheduler.step 返回 CPU tensor）
        # 导致 decoders.py line 185 报 "Expected all tensors to be on the same device" 错误
        # hook MiniMaxH3VideoDecodeStep / MiniMaxH3AudioDecodeStep，确保 latents 在 GPU 上
        self._patch_decoders_device()

        # v0.6.21+ 修复：VAE 编码触发 `aten::slow_conv3d_forward` CUDA 后端缺失
        # （cuDNN 缺该路径，只 CPU 后端有），导致 I2V/FL2V/Ref2V/video_edit 全部失败。
        # patch encode_vae_condition 让 vae.encode 在 CPU 上跑（结果本来就在 CPU）
        self._patch_encoders_device()

    def _load_all_to_gpu(self) -> None:
        """L0 策略：全部加载到 GPU（需 ≥48GB 显存）。

        所有组件常驻 GPU，推理速度最快。

        v0.6.9+ 修订：添加显存校验，确保模型真正移到 GPU。
        """
        self._logger.info("[L0] 全部加载到 GPU（不卸载）")
        if self._device.startswith("cuda"):
            vram_before_mb = self._get_gpu_memory_used_mb()
            try:
                self._pipe.to(self._device)
                self._logger.info("[L0] .to(%s) 成功", self._device)
            except Exception as e:
                self._logger.warning("[L0] .to(%s) 失败: %s", self._device, e)
            # 显存校验
            vram_after_mb = self._get_gpu_memory_used_mb()
            if vram_before_mb is not None and vram_after_mb is not None:
                delta_mb = vram_after_mb - vram_before_mb
                self._logger.info(
                    "[L0] GPU 显存变化: %dMB → %dMB (Δ%dMB)",
                    vram_before_mb, vram_after_mb, delta_mb,
                )
                if delta_mb < 500:
                    self._logger.error(
                        "[L0] ⚠ GPU 显存仅增加 %dMB，模型可能未真正加载到 GPU！", delta_mb,
                    )

    def _load_with_te_offload(self) -> None:
        """L1 策略：轻度卸载（TE 编码后移到 CPU）。

        text_encoder 在编码完 prompt 后移到 CPU，释放显存给 transformer。
        使用 diffusers 的 enable_model_cpu_offload（组件级 offload）。

        v0.6.9+ 修订：ModularPipeline 不支持 enable_model_cpu_offload，
        失败时 fallback 到 .to(device)，并添加显存校验。

        v0.6.11+ 修订：配合 _resolve_sdnq_load_device 的 CPU Offload 策略，
        text_encoder 已在 SDNQ 加载阶段加载到 CPU。本方法只需移动
        transformer/vae/audio_vae 到 GPU（与 L2 相同的逐组件移动逻辑）。
        """
        self._logger.info("[L1] 轻度卸载（逐组件移动到 GPU，text_encoder 保持 CPU）")
        vram_before_mb = self._get_gpu_memory_used_mb()

        if not self._device.startswith("cuda"):
            self._logger.info("[L1] device=%s 非 CUDA，跳过移动", self._device)
            return

        import os
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # v0.6.11+ text_encoder 已加载到 CPU，保持不动
        cpu_offload_components = {"text_encoder"}

        import torch
        moved_count = 0
        skipped_count = 0
        failed_count = 0
        for name, component in self._pipe.components.items():
            if not hasattr(component, "parameters") and not hasattr(component, "to"):
                skipped_count += 1
                continue
            if name in cpu_offload_components:
                skipped_count += 1
                self._logger.info(
                    "[L1] 组件 %s 保持 CPU（CPU Offload 策略）",
                    name,
                )
                continue
            try:
                if hasattr(component, "device"):
                    try:
                        curr_device = component.device
                        if str(curr_device) == self._device:
                            skipped_count += 1
                            continue
                    except Exception:
                        pass
                component.to(self._device)
                moved_count += 1
                vram_now = self._get_gpu_memory_used_mb()
                self._logger.info(
                    "[L1] 组件 %s 已移到 %s（GPU 显存: %sMB）",
                    name, self._device, vram_now,
                )
            except Exception as e:
                failed_count += 1
                self._logger.warning(
                    "[L1] 组件 %s 移动到 GPU 失败（保持 CPU）: %s", name, e,
                )

        self._logger.info(
            "[L1] 逐组件移动完成: 成功=%d, 跳过=%d, 失败=%d",
            moved_count, skipped_count, failed_count,
        )
        vram_after_mb = self._get_gpu_memory_used_mb()
        if vram_before_mb is not None and vram_after_mb is not None:
            delta_mb = vram_after_mb - vram_before_mb
            self._logger.info(
                "[L1] GPU 显存变化: %dMB → %dMB (Δ%dMB)",
                vram_before_mb, vram_after_mb, delta_mb,
            )
            if delta_mb < 500:
                self._logger.error(
                    "[L1] ⚠ GPU 显存仅增加 %dMB，模型可能未真正加载到 GPU！", delta_mb,
                )

    def _load_with_layer_offload(self) -> None:
        """L2 策略：中度卸载（组件级 offload）。

        v0.5.1+ 修订：原实现调用 enable_sequential_cpu_offload（层级 offload），
        对 60GB+ 模型会逐层注册 hook，耗时 10-30 分钟且全程持有 GIL，
        导致事件循环阻塞、健康检查无响应。

        新实现改用 enable_model_cpu_offload（组件级 offload），仅注册组件级
        hook（数量少，耗时秒级），推理时整个组件搬入/搬出 GPU。
        显存占用与 L1 相同，但接受显存不足时 OOM 风险（推理时降级处理）。

        v0.6.9+ 修订：ModularPipeline 类**没有** enable_model_cpu_offload 方法
        （只有 DiffusionPipeline 才有）。.to(device) 一次性移动所有组件时
        因显存碎片化导致 OOM（只差 64MB）。修复：逐组件移动，跳过已在 GPU
        的组件，每次移动后释放 CPU 副本；同时通过 nvidia-smi 校验显存增长。

        v0.6.11+ 修订：配合 _resolve_sdnq_load_device 的 CPU Offload 策略，
        text_encoder 已在 SDNQ 加载阶段加载到 CPU（释放 ~19.6GB 显存）。
        本方法只需移动 transformer/vae/audio_vae 到 GPU，跳过已在 CPU 的
        text_encoder（避免把 19.6GB 模型重新搬回 GPU 导致 OOM）。
        预期 GPU 显存占用：transformer(17.5GB) + vae(5.8GB) ≈ 23GB，
        剩余 ~25GB 用于 attention 中间张量。
        """
        self._logger.info("[L2] 中度卸载（逐组件移动到 GPU，text_encoder 保持 CPU）")
        # 记录加载前 GPU 显存（用于验证模型是否真的移到 GPU）
        vram_before_mb = self._get_gpu_memory_used_mb()

        if not self._device.startswith("cuda"):
            self._logger.info("[L2] device=%s 非 CUDA，跳过移动", self._device)
            return

        # 设置 expandable_segments 减少碎片化（如果尚未设置）
        import os
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            self._logger.info("[L2] 已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

        # v0.6.11+ CPU Offload 策略：text_encoder 已加载到 CPU，保持不动
        # 这样 GPU 只承载 transformer + vae + audio_vae ≈ 23GB，留出 25GB 给 attention
        cpu_offload_components = {"text_encoder"}  # 这些组件保持 CPU

        # 逐组件移动到 GPU（避免一次性 .to() 导致 OOM）
        import torch
        moved_count = 0
        skipped_count = 0
        failed_count = 0
        for name, component in self._pipe.components.items():
            # 跳过非模型组件（tokenizer/processor/scheduler 等无参数）
            if not hasattr(component, "parameters") and not hasattr(component, "to"):
                skipped_count += 1
                continue
            # v0.6.11+ 跳过 CPU Offload 组件（text_encoder 保持 CPU）
            if name in cpu_offload_components:
                skipped_count += 1
                self._logger.info(
                    "[L2] 组件 %s 保持 CPU（CPU Offload 策略，释放显存给 attention）",
                    name,
                )
                continue
            try:
                # 检查组件当前设备
                if hasattr(component, "device"):
                    try:
                        curr_device = component.device
                        if str(curr_device) == self._device:
                            skipped_count += 1
                            continue
                    except Exception:
                        pass
                # 逐组件移动
                component.to(self._device)
                moved_count += 1
                vram_now = self._get_gpu_memory_used_mb()
                self._logger.info(
                    "[L2] 组件 %s 已移到 %s（GPU 显存: %sMB）",
                    name, self._device, vram_now,
                )
            except Exception as e:
                failed_count += 1
                self._logger.warning(
                    "[L2] 组件 %s 移动到 GPU 失败（保持 CPU）: %s", name, e,
                )

        self._logger.info(
            "[L2] 逐组件移动完成: 成功=%d, 跳过=%d, 失败=%d",
            moved_count, skipped_count, failed_count,
        )

        # 校验：GPU 显存是否显著增加
        vram_after_mb = self._get_gpu_memory_used_mb()
        if vram_before_mb is not None and vram_after_mb is not None:
            delta_mb = vram_after_mb - vram_before_mb
            self._logger.info(
                "[L2] GPU 显存变化: %dMB → %dMB (Δ%dMB)",
                vram_before_mb, vram_after_mb, delta_mb,
            )
            if delta_mb < 500:
                self._logger.error(
                    "[L2] ⚠ GPU 显存仅增加 %dMB（< 500MB），模型可能未真正加载到 GPU！",
                    delta_mb,
                )
            else:
                self._logger.info(
                    "[L2] ✓ GPU 显存增加 %dMB，模型已成功加载到 GPU", delta_mb,
                )

    def _load_with_aggressive_offload(self) -> None:
        """L3 策略：重度卸载（模型保持 CPU，推理时按需移到 GPU）。

        v0.5.1+ 修订：原实现调用 enable_sequential_cpu_offload，对 60GB+
        模型极慢。新实现不做任何 offload，让模型保持在 load_components
        后的默认状态（CPU，safetensors mmap）。

        v0.6.11+ 修订：配合 _resolve_sdnq_load_device 的 CPU Offload 策略，
        text_encoder 和 transformer 都已加载到 CPU（用 RAM）。
        本方法记录 L3 状态，推理时由 generate() 方法按需把 transformer
        移到 GPU（推理必需），其他组件（text_encoder/vae）保持 CPU。

        优点：显存占用极低（仅推理时临时占用 transformer 17.5GB）
        缺点：推理速度慢（每次推理需要移动 transformer，且 text_encoder 在 CPU 编码）

        注意：此模式下推理速度极慢，仅用于显存严重不足（< 24GB）的场景。
        """
        self._logger.info(
            "[L3] 重度卸载（模型保持 CPU，推理时按需把 transformer 移到 GPU）",
        )
        # 不调用任何 offload 方法，模型保持在 CPU
        # 推理时由 generate() 方法手动管理设备移动
        vram_after_mb = self._get_gpu_memory_used_mb()
        self._logger.info(
            "[L3] 加载完成 GPU 显存占用: %sMB（模型在 RAM，GPU 几乎不占显存）",
            vram_after_mb,
        )

    # ============ 推理 ============

    def generate(self, mode: str, prompt: str, **kwargs: Any) -> dict:
        """同步生成视频（含可选音频）—— 真实推理版。

        基于 diffusers ModularPipeline 调用 MiniMax-H3 模型生成视频。

        Args:
            mode: 生成模式：
                - "t2v":         Text to Video（纯文本生成视频，无音频）
                - "i2v":         Image to Video（图片 + 文本生成视频，无音频）
                - "fl2v":        Full Latent to Video（首帧 + 文本生成含音频视频）
                - "ref2v":       Reference to Video（参考图 + 文本生成含音频视频）
                - "ref2va":      多模态参考（图/视频/音频混合参考 + 文本生成含音频视频）
                - "video_edit":  视频编辑（参考视频 + 文本生成视频）
            prompt: 提示词（已通过 PromptProcessor 处理）
            **kwargs: 其他推理参数：
                - negative_prompt: 负向提示词（MiniMax-H3 不支持，忽略）
                - duration: 时长（秒），5-15
                - fps: 帧率（MiniMax-H3 固定 24fps，其他值忽略）
                - width / height: 分辨率（必须为 32 的倍数）
                - num_inference_steps: 推理步数
                - guidance_scale: CFG 引导强度（MiniMax-H3 guidance-distilled，忽略）
                - seed: 随机种子（-1=随机）
                - image: 参考图片路径/URL（I2V / Ref2VA 模式）
                - video: 参考视频路径/URL（video_edit 模式）
                - reference_images: 图片参考路径列表（ref2va 模式，≤9）
                - reference_videos: 视频参考路径列表（ref2va 模式，≤3）
                - reference_audios: 音频参考路径列表（ref2va 模式，≤3）

        Returns:
            dict 含：
                - video_frames: 视频帧序列 (T, H, W, C) uint8
                - audio_data: 音频数据 (channels, samples) float32（无音频时为 None）
                - metadata: 元数据字典（mode / seed / steps / 等）

        Raises:
            RuntimeError: 模型未加载 / 推理失败
        """
        with self._lock:
            if self._pipe is None:
                raise RuntimeError("模型未加载，请先调用 load()")

            import numpy as np
            import torch

            start_time = time.time()

            # 提取推理参数
            negative_prompt = kwargs.get("negative_prompt", "")
            duration = kwargs.get("duration", 5)
            fps = 24  # MiniMax-H3 固定 24fps（忽略传入的 fps）
            width = kwargs.get("width", 1344)
            height = kwargs.get("height", 768)
            num_steps = kwargs.get("num_inference_steps", 30)
            guidance = kwargs.get("guidance_scale", 7.0)  # MiniMax-H3 guidance-distilled，忽略
            seed = kwargs.get("seed", -1)
            # v1.1+ sigma shift（对齐官方 turbo 蒸馏参数）
            # diffusers MiniMaxH3Scheduler 固定 Euler + 指数 shift，每请求经 set_shift 覆盖：
            #   官方 8 步 turbo shift=12/3（视频/音频），4 步 768p 版 shift=6/3
            flow_shift = kwargs.get("flow_shift")
            audio_flow_shift = kwargs.get("audio_flow_shift")
            image_input = kwargs.get("image")
            last_image_input = kwargs.get("last_image")
            references_input = kwargs.get("references")
            video_input = kwargs.get("video")

            # v0.7.7+ T2.2 conditioning 缓存 key：
            # 相同（variant + mode + prompt + 参考输入指纹）→ conditioning 相同，
            # TE 编码 patch 命中缓存时跳过 19.6GB TE 的编码（含 L2 临时上 GPU）。
            # 参考输入指纹 = 路径 + 大小 + mtime（读文件内容 hash 对视频过大，且
            # mtime/size 已能覆盖实际使用中的替换场景）。
            _cond_components: list[Any] = [
                self._loaded_variant or "?", mode, prompt,
                f"WxH:{int(width)}x{int(height)}", f"dur:{float(duration)}",
            ]
            for _cn, _cv in (
                ("image", image_input),
                ("last_image", last_image_input),
                ("video", video_input),
            ):
                if _cv:
                    _cp = Path(_cv)
                    try:
                        _cst = _cp.stat()
                        _cond_components.append(f"{_cn}:{_cp}:{_cst.st_size}:{int(_cst.st_mtime)}")
                    except OSError:
                        _cond_components.append(f"{_cn}:{_cv}")
            for _ref in (references_input or []):
                _rp = Path(_ref)
                try:
                    _rst = _rp.stat()
                    _cond_components.append(f"ref:{_rp}:{_rst.st_size}:{int(_rst.st_mtime)}")
                except OSError:
                    _cond_components.append(f"ref:{_ref}")
            self._cond_cache_key = (
                self._cond_cache.make_key(*_cond_components)
                if self._cond_cache is not None else None
            )

            # 时长范围约束（MiniMax-H3: 5-15s）
            duration = max(5.0, min(15.0, float(duration)))

            self._logger.info(
                "MiniMax-H3 生成: mode=%s, prompt=%r, duration=%.1fs, size=%dx%d, steps=%d, seed=%d",
                mode,
                prompt[:50] + ("..." if len(prompt) > 50 else ""),
                duration, height, width, num_steps, seed,
            )

            # v1.2+（41 号任务）竖屏发散预警：
            # 实测 2026-09-16 竖屏（480x864 / 416x736）+ fp16 + SDNQ 组合下扩散 latents 必现 NaN→灰屏。
            # 方案默认均为横屏 864x480（已知稳定）。极端竖屏长宽比在低精度下易数值发散，
            # 提前告警并给出现成替代方案，避免用户再跑几分钟拿到灰屏/失败。
            _eps = 1e-6
            _ar = max(width, height) / (min(width, height) + _eps)
            if (height * 1.6) > width and width > 0:  # 竖屏且长宽比 > 1.6:1
                self._logger.warning(
                    "[MiniMax-H3 竖屏预警] 当前为强竖屏 %dx%d（长宽比 %.2f:1）。"
                    "fp16+SDNQ 下竖屏存在扩散 NaN/灰屏风险（方案默认横屏 864x480 最稳）。"
                    "若生成失败或灰屏，请改用横屏尺寸（如 864x480 / 768x432）",
                    width, height, _ar,
                )

            # 处理 seed
            if seed == -1:
                generator = torch.Generator("cpu")
                self._logger.info("使用随机 seed")
            else:
                generator = torch.Generator("cpu").manual_seed(int(seed))
                self._logger.info("使用固定 seed=%d", seed)

            # 计算帧数：duration * 24fps（pipeline 内部会自动对齐到 17n+5）
            num_frames = max(1, int(round(duration * fps)))

            # 构建推理参数
            pipe_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": num_steps,
                "generator": generator,
                "output_type": "np",  # 输出 numpy 数组，形状 (1, T, H, W, C) [0,1]
            }

            # ----- 多模态输入处理 -----
            self._logger.info(
                "准备多模态输入: mode=%s, has_image=%s, has_last_image=%s, has_references=%s, has_video=%s",
                mode, image_input is not None, last_image_input is not None,
                references_input is not None, video_input is not None,
            )

            if mode == "t2v":
                # T2V：纯文本生成视频，不需要任何图片输入
                # workflow="t2va" 的条件块会自动选择 t2va 分支
                self._logger.info("T2V 模式：纯文本生成（无需图片）")

            elif mode == "i2v" and image_input:
                # I2V：使用 image 作为首帧
                pipe_kwargs["image"] = self._load_image(image_input)
                self._logger.info("已加载首帧图片（I2V）: %s", image_input)

            elif mode == "fl2v" and image_input:
                # FL2V：首帧 + 可选尾帧
                # MiniMax-H3 fl2va workflow 接受 image（首帧）和 last_image（尾帧）
                pipe_kwargs["image"] = self._load_image(image_input)
                self._logger.info("已加载首帧图片（FL2V）: %s", image_input)
                if last_image_input:
                    pipe_kwargs["last_image"] = self._load_image(last_image_input)
                    self._logger.info("已加载尾帧图片（FL2V）: %s", last_image_input)

            elif mode == "ref2v" and (references_input or image_input):
                # Ref2V：多张参考图 + 文本生成含音频视频
                # 使用 references 列表（MiniMaxH3ImageReference）
                try:
                    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
                except ImportError:
                    # fallback：直接传 image 参数（让 pipeline 自动处理）
                    self._logger.warning(
                        "无法导入 MiniMaxH3ImageReference，退回直接传 image 参数",
                    )
                    if image_input:
                        pipe_kwargs["image"] = self._load_image(image_input)
                else:
                    # 收集所有参考图片路径
                    ref_paths = references_input if references_input else ([image_input] if image_input else [])
                    ref_list = []
                    for i, ref_path in enumerate(ref_paths[:9]):  # 最多 9 张
                        try:
                            ref_img = self._load_image(ref_path)
                            # v0.7.7 方案3：封顶参考图视觉 token 数，降压长 token 编码
                            ref_img = self._cap_reference_vision_tokens(ref_img)
                            ref_list.append(MiniMaxH3ImageReference(image=ref_img))
                            self._logger.info("已加载参考图 %d（Ref2V）: %s", i + 1, ref_path)
                        except Exception as e:
                            self._logger.warning("参考图 %d 加载失败，跳过: %s -> %s", i + 1, ref_path, e)
                    if ref_list:
                        pipe_kwargs["references"] = ref_list
                        self._logger.info("Ref2V 共加载 %d 张参考图", len(ref_list))

            elif mode == "ref2va":
                # Ref2VA 多模态参考：图片/视频/音频混合参考 + 文本生成含音频视频
                # 官方约定：references 列表顺序决定 prompt 中 <Picture i>/<Video k>/<Audio j>
                # 标签编号与 rotary clock 排布——"不同顺序就是不同请求"。
                # 引擎侧固定顺序：图片（按上传序）→ 视频（按上传序）→ 音频（按上传序），
                # 与 prompt_processor 的 Context-IR 标签构造顺序保持一致。
                ref_images = kwargs.get("reference_images") or []
                ref_videos = kwargs.get("reference_videos") or []
                ref_audios = kwargs.get("reference_audios") or []

                if not (ref_images or ref_videos or ref_audios):
                    self._logger.warning(
                        "ref2va 模式未收到任何参考输入（reference_images/videos/audios 均为空）"
                    )
                else:
                    try:
                        from diffusers.modular_pipelines.minimax_h3 import (
                            MiniMaxH3AudioReference,
                            MiniMaxH3ImageReference,
                            MiniMaxH3VideoReference,
                        )
                    except ImportError as e:
                        raise RuntimeError(
                            "当前 diffusers 版本不支持 MiniMax-H3 多模态参考"
                            "（MiniMaxH3ImageReference / VideoReference / AudioReference "
                            f"不可导入）: {e}"
                        ) from e

                    ref_list = []
                    # 1. 图片参考（≤9，按上传序）
                    for i, ref_path in enumerate(ref_images[:9]):
                        try:
                            ref_list.append(
                                MiniMaxH3ImageReference(
                                    image=self._cap_reference_vision_tokens(
                                        self._load_image(ref_path)
                                    )
                                )
                            )
                            self._logger.info(
                                "已加载参考图 %d（Ref2VA）: %s", i + 1, ref_path
                            )
                        except Exception as e:
                            self._logger.warning(
                                "参考图 %d 加载失败，跳过: %s -> %s", i + 1, ref_path, e
                            )
                    # 2. 视频参考（≤3，官方 from_file：PyAV 解码，保留真实帧率与音轨）
                    for k, ref_path in enumerate(ref_videos[:3]):
                        try:
                            ref_list.append(MiniMaxH3VideoReference.from_file(ref_path))
                            self._logger.info(
                                "已加载参考视频 %d（Ref2VA）: %s", k + 1, ref_path
                            )
                        except Exception as e:
                            self._logger.warning(
                                "参考视频 %d 加载失败，跳过: %s -> %s", k + 1, ref_path, e
                            )
                    # 3. 音频参考（≤3，官方 from_file：PyAV 解码，支持音频文件或视频音轨）
                    #    官方硬约束：音频参考不能单独使用，engine.py 入口已校验
                    for j, ref_path in enumerate(ref_audios[:3]):
                        try:
                            ref_list.append(MiniMaxH3AudioReference.from_file(ref_path))
                            self._logger.info(
                                "已加载参考音频 %d（Ref2VA）: %s", j + 1, ref_path
                            )
                        except Exception as e:
                            self._logger.warning(
                                "参考音频 %d 加载失败，跳过: %s -> %s", j + 1, ref_path, e
                            )

                    if ref_list:
                        pipe_kwargs["references"] = ref_list
                        # 顺序语义日志（R8：标签编号与 references 顺序一致性排查依据）
                        self._logger.info(
                            "Ref2VA 共加载 %d 个参考（图 %d / 视频 %d / 音频 %d），"
                            "references 顺序: %s",
                            len(ref_list), len(ref_images[:9]), len(ref_videos[:3]),
                            len(ref_audios[:3]),
                            " ".join(
                                f"<{type(r).__name__.replace('MiniMaxH3', '').replace('Reference', '')} {idx}>"
                                for idx, r in enumerate(ref_list, start=1)
                            ),
                        )
                    else:
                        # 入口校验已保证至少 1 个参考路径，走到这里说明全部加载失败
                        raise RuntimeError(
                            "ref2va 模式所有参考文件均加载失败（图 %d / 视频 %d / "
                            "音频 %d），请检查文件格式与完整性后重试"
                            % (len(ref_images), len(ref_videos), len(ref_audios))
                        )

            elif mode == "video_edit" and video_input:
                # VideoEdit：参考视频 + 文本生成视频
                # MiniMaxH3VideoReference(frames: list[PIL.Image], fps, audio, sample_rate)
                # 接受的是 PIL.Image 帧列表 + fps，不是 video 路径
                try:
                    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
                except ImportError:
                    self._logger.warning(
                        "无法导入 MiniMaxH3VideoReference，video_edit 模式可能失败",
                    )
                else:
                    frames, src_fps = self._load_video_frames(video_input, max_frames=49)
                    pipe_kwargs["references"] = [
                        MiniMaxH3VideoReference(frames=frames, fps=src_fps)
                    ]
                    self._logger.info(
                        "已加载参考视频: %s (frames=%d, fps=%.2f)",
                        video_input, len(frames), src_fps,
                    )

            # ----- 执行推理 -----
            # v0.6.11+ CPU Offload 策略：根据 offload_level 管理设备移动
            #
            # L0（全部 GPU）：
            #   - text_encoder 在 GPU，推理前卸载到 CPU（释放 ~19.6GB 给 attention）
            #   - transformer 在 GPU，推理后保持 GPU
            #   - 推理后恢复 text_encoder 到 GPU
            #
            # L1/L2（text_encoder 在 CPU，transformer 在 GPU）：
            #   - text_encoder 已在 CPU（加载阶段决定），无需移动
            #   - transformer 在 GPU，直接推理
            #   - GPU 显存占用：transformer(17.5GB) + vae(5.8GB) ≈ 23GB，剩余 25GB 给 attention
            #
            # L3（全部 CPU，推理时按需移动）：
            #   - text_encoder 在 CPU（prompt 在 CPU 编码）
            #   - transformer 推理前移到 GPU，推理后移回 CPU（释放显存）
            #   - vae 推理前移到 GPU，推理后移回 CPU
            #   - GPU 显存占用：推理时 ~23GB，空闲时 ~0GB

            _te_offloaded = False
            _transformer_moved_to_gpu = False
            _vae_moved_to_gpu = False
            _te_l2_moved_to_gpu = False  # L2: text_encoder 临时上 GPU
            _te_l2_hook_handle = None    # L2: forward hook handle

            # L2 模式：text_encoder 在 CPU，推理前临时移到 GPU 加速 prompt encoding
            # 通过 forward hook 在 prompt encoding 完成后自动移回 CPU（释放显存给 transformer）
            # 注意：text_encoder ~19.6GB，需要先卸载 vae/audio_vae ~5.8GB 腾出显存
            #
            # v0.6.13+ 修订：修复 hook 注册位置
            #   - 之前 hook 注册在 text_encoder 上，但 encoders.py 调用 text_encoder.model(...)
            #   - 导致 hook 不触发，TE 一直留在 GPU，扩散推理时 OOM
            #   - 现在 hook 注册在 text_encoder.model 上，TE forward 完成后立即移回 CPU
            #   - TE 编码时 GPU 显存 = transformer(17.5GB) + TE(19.6GB) ≈ 37GB（vae 已下 CPU）
            #   - TE 编码后 GPU 显存 = transformer(17.5GB) + vae(5.8GB) + prompt_embeds(1GB) ≈ 24GB
            #   - 剩余 25GB 给 attention，不会 OOM
            #
            # v0.6.14+ 修订：L3 模式不再让 TE 临时上 GPU
            #   - L2 模式下 TE 上 GPU 时峰值 39GB，扩散 attention 仍可能 OOM（48GB 显存不够）
            #   - L3 模式：TE 保持在 CPU 编码（用 RAM，系统有 64GB RAM 足够）
            #   - prompt_embeds 编码后由 _patched_call（denoise.py patch）移到 GPU
            #   - GPU 显存需求：transformer(17.5GB) + vae(5.8GB) + attention(10-15GB) ≈ 33-38GB < 48GB
            #   - 代价：CPU 编码 prompt 较慢（~30s），但避免 OOM 崩溃
            _l2_vae_offloaded = []  # 记录被卸载的 vae 组件名
            # v0.6.16+ 禁用 L2 的 TE 临时上 GPU（hook 不可靠导致 TE 留在 GPU → OOM）
            # L2 现在改为 TE 一直在 CPU 编码（与 L1 一致），通过 _patch_prompt_embeds_device 移动 prompt_embeds
            if False and self._offload_level == "L2" and self._device.startswith("cuda"):
                try:
                    if hasattr(self._pipe, "text_encoder"):
                        te = self._pipe.text_encoder
                        if hasattr(te, "device") and not str(te.device).startswith("cuda"):
                            # 1. 先卸载 vae/audio_vae 到 CPU（腾出显存给 text_encoder）
                            for vae_name in ("vae", "audio_vae"):
                                try:
                                    vae_comp = getattr(self._pipe, vae_name, None)
                                    if (vae_comp is not None and hasattr(vae_comp, "device")
                                            and str(vae_comp.device).startswith("cuda")):
                                        vae_comp.to("cpu")
                                        _l2_vae_offloaded.append(vae_name)
                                        self._logger.info(
                                            "[%s VAE 临时下 CPU] %s 已移到 CPU（腾显存给 text_encoder）",
                                            self._offload_level, vae_name,
                                        )
                                except Exception as _e:
                                    self._logger.warning(
                                        "[L2 VAE 临时下 CPU] %s 卸载失败: %s", vae_name, _e)
                            if _l2_vae_offloaded:
                                torch.cuda.empty_cache()

                            # 2. 加载 text_encoder 到 GPU
                            vram_before = self._get_gpu_memory_used_mb()
                            te.to(self._device)
                            torch.cuda.empty_cache()
                            vram_after = self._get_gpu_memory_used_mb()
                            self._logger.info(
                                "[L2 TE 临时上 GPU] text_encoder 已移到 GPU（GPU 显存: %sMB → %sMB），"
                                "prompt encoding 完成后自动移回 CPU 并恢复 VAE",
                                vram_before, vram_after,
                            )
                            _te_l2_moved_to_gpu = True

                            # 3. 注册 forward hook：text_encoder forward 完成后
                            #    a. 移回 CPU（释放 ~19.6GB）— 移动整个 te，不只是 module(te.model)
                            #    b. 恢复 vae/audio_vae 到 GPU（推理需要）
                            def _te_post_forward_hook(module, _input, _output):
                                nonlocal _te_l2_hook_handle
                                try:
                                    vram_before_hook = self._get_gpu_memory_used_mb()
                                    # v0.6.13+ 修复：移动整个 text_encoder 到 CPU
                                    # module 是 te.model，但 te 可能还有 lm_head 等组件在 GPU
                                    te.to("cpu")
                                    torch.cuda.empty_cache()
                                    vram_after_hook = self._get_gpu_memory_used_mb()
                                    freed = (vram_before_hook - vram_after_hook
                                             if vram_before_hook and vram_after_hook else 0)
                                    self._logger.info(
                                        "[L2 TE Auto Offload] text_encoder forward 完成，"
                                        "已移回 CPU（释放 %sMB 显存）",
                                        freed,
                                    )
                                    # 恢复 vae/audio_vae 到 GPU
                                    for vae_name in _l2_vae_offloaded:
                                        try:
                                            vae_comp = getattr(self._pipe, vae_name, None)
                                            if vae_comp is not None:
                                                vae_comp.to(self._device)
                                                self._logger.info(
                                                    "[L2 VAE 恢复] %s 已移回 GPU", vae_name)
                                        except Exception as _e:
                                            self._logger.warning(
                                                "[L2 VAE 恢复] %s 移回 GPU 失败: %s",
                                                vae_name, _e)
                                    torch.cuda.empty_cache()
                                    # 移除 hook（只执行一次）
                                    if _te_l2_hook_handle is not None:
                                        _te_l2_hook_handle.remove()
                                        _te_l2_hook_handle = None
                                except Exception as _e:
                                    self._logger.warning(
                                        "[L2 TE Auto Offload] text_encoder 移回 CPU 失败: %s", _e)

                            # v0.6.13+ 修复：hook 注册在 text_encoder.model 上
                            # encoders.py 调用 text_encoder.model(...)，不是 text_encoder(...)
                            # 如果 hook 注册在 te 上，不会触发，TE 会一直留在 GPU 导致 OOM
                            _hook_target = te.model if hasattr(te, "model") else te
                            _te_l2_hook_handle = _hook_target.register_forward_hook(
                                _te_post_forward_hook
                            )
                            self._logger.info(
                                "[L2 TE Hook] hook 已注册在 %s（te.model 存在: %s）",
                                "te.model" if hasattr(te, "model") else "te",
                                hasattr(te, "model"),
                            )
                except Exception as _e:
                    self._logger.warning("[L2 TE 临时上 GPU] 失败: %s", _e)
                    # 失败时恢复 vae 到 GPU
                    for vae_name in _l2_vae_offloaded:
                        try:
                            vae_comp = getattr(self._pipe, vae_name, None)
                            if vae_comp is not None:
                                vae_comp.to(self._device)
                                self._logger.info(
                                    "[L2 VAE 恢复（失败回退）] %s 已移回 GPU", vae_name)
                        except Exception:
                            pass

            # L0 模式：text_encoder 在 GPU，推理前卸载到 CPU
            if self._offload_level == "L0" and self._device.startswith("cuda"):
                try:
                    if hasattr(self._pipe, "text_encoder"):
                        te = self._pipe.text_encoder
                        if hasattr(te, "device") and str(te.device).startswith("cuda"):
                            vram_before = self._get_gpu_memory_used_mb()
                            te.to("cpu")
                            torch.cuda.empty_cache()
                            vram_after = self._get_gpu_memory_used_mb()
                            freed = vram_before - vram_after if vram_before and vram_after else 0
                            self._logger.info(
                                "[L0 TE Offload] text_encoder 已卸载到 CPU（释放 %sMB 显存）",
                                freed,
                            )
                            _te_offloaded = True
                except Exception as _e:
                    self._logger.warning("[L0 TE Offload] text_encoder 卸载失败（忽略）: %s", _e)

            # L3 模式：transformer 和 vae 在 CPU，推理前移到 GPU
            if self._offload_level == "L3" and self._device.startswith("cuda"):
                # v0.6.12+ 修复：用 paradigm 判断 transformer_attr（hasattr 不可靠，
                # transformer_ref 可能存在但为 None，导致 tr is None 跳过移动且无日志）
                try:
                    _paradigm = (self._detect_paradigm(self._loaded_variant)
                                 if self._loaded_variant else "FL2VA")
                except Exception:
                    _paradigm = "FL2VA"
                transformer_attr = "transformer" if _paradigm == "FL2VA" else "transformer_ref"
                try:
                    tr = getattr(self._pipe, transformer_attr, None)
                    if tr is not None:
                        vram_before = self._get_gpu_memory_used_mb()
                        tr.to(self._device)
                        torch.cuda.empty_cache()
                        vram_after = self._get_gpu_memory_used_mb()
                        self._logger.info(
                            "[L3] %s 已移到 GPU（paradigm=%s, GPU 显存: %sMB → %sMB）",
                            transformer_attr, _paradigm, vram_before, vram_after,
                        )
                        _transformer_moved_to_gpu = True
                    else:
                        self._logger.error(
                            "[L3] %s 为 None（paradigm=%s），无法移动到 GPU！"
                            "transformer 将保持在 CPU 推理（极慢）",
                            transformer_attr, _paradigm,
                        )
                except Exception as _e:
                    self._logger.warning("[L3] transformer 移动到 GPU 失败: %s", _e)

                # 移动 vae/audio_vae 到 GPU（VAE 解码需要 GPU 加速）
                for vae_name in ("vae", "audio_vae"):
                    try:
                        vae_comp = getattr(self._pipe, vae_name, None)
                        if vae_comp is not None and hasattr(vae_comp, "device"):
                            if not str(vae_comp.device).startswith("cuda"):
                                vae_comp.to(self._device)
                                self._logger.info("[L3] %s 已移到 GPU", vae_name)
                                _vae_moved_to_gpu = True
                    except Exception as _e:
                        self._logger.warning("[L3] %s 移动到 GPU 失败: %s", vae_name, _e)

            # v0.7.7+ T2.1 推理前空闲显存日志（排查 OOM / "运行不了"）
            try:
                import torch as _t2
                if _t2.cuda.is_available():
                    _f2_b, _t2_b = _t2.cuda.mem_get_info()
                    self._logger.info(
                        "[T2.1 推理前显存] free=%.2fGB / total=%.2fGB（offload=%s）",
                        _f2_b / (1024 ** 3), _t2_b / (1024 ** 3), self._offload_level,
                    )
            except Exception as _e2:
                self._logger.debug("[T2.1 推理前显存] 查询失败: %s", _e2)

            # v1.1+ sigma shift 生效：diffusers MiniMaxH3Scheduler 固定 Euler + 指数 shift，
            # 每请求经 set_shift 覆盖（必须在 pipeline 内部 set_timesteps 前调用）。
            # 官方 turbo：8 步 shift=12/3（视频/音频），4 步 768p 版 shift=6/3。
            if flow_shift is not None or audio_flow_shift is not None:
                try:
                    for _sched_name, _sched, _shift in (
                        ("scheduler", getattr(self._pipe, "scheduler", None), flow_shift),
                        ("audio_scheduler", getattr(self._pipe, "audio_scheduler", None), audio_flow_shift),
                    ):
                        if _sched is None or _shift is None:
                            continue
                        if hasattr(_sched, "set_shift"):
                            _sched.set_shift(float(_shift))
                            self._logger.info(
                                "[shift] %s set_shift=%.2f（类 %s）",
                                _sched_name, float(_shift), type(_sched).__name__,
                            )
                        else:
                            self._logger.warning(
                                "[shift] %s（类 %s）不支持 set_shift，sigma shift 未生效",
                                _sched_name, type(_sched).__name__,
                            )
                except Exception as _e:
                    self._logger.warning("[shift] 设置 sigma shift 失败: %s", _e)

            self._logger.info(
                "开始扩散推理（这可能需要几分钟）[offload=%s]",
                self._offload_level,
            )
            # v0.7.9+ 复线：在 pipeline 调用前，对 before_encoder 实际读取的
            # `components.config` 强制覆写 reference_image_short_edge。
            # 行 314 的 set 在 load_components / SDNQ 装载后会被重置回 2048，
            # 导致 ref2v/ref2va 仍按短边 2048 放大编码（3 图→21621 token→CPU 卡死）。
            # 这里所有候选 config 对象都覆写一遍，并读回实际值打日志便于确认命中哪个。
            _ref_short_edge = 768
            _hit_whom = []
            for _pc in (self._pipe, getattr(self._pipe, "components", None)):
                _c = getattr(_pc, "config", None)
                if _c is not None and hasattr(_c, "reference_image_short_edge"):
                    _c.reference_image_short_edge = _ref_short_edge
                    _hit_whom.append(type(_pc).__name__)
                _components_cfg = getattr(_pc, "components", None)
                _cc = getattr(_components_cfg, "config", None)
                if _cc is not None and hasattr(_cc, "reference_image_short_edge"):
                    _cc.reference_image_short_edge = _ref_short_edge
                    _hit_whom.append(type(_pc).__name__ + ".components")
            self._logger.warning(
                "[ref-pixel] reference_image_short_edge=%s 命中对象: %s（pipe=%s, comp=%s）",
                _ref_short_edge,
                _hit_whom or "无候选",
                getattr(self._pipe.config, "reference_image_short_edge", "NA"),
                getattr(
                    getattr(getattr(self._pipe, "components", None), "config", None),
                    "reference_image_short_edge",
                    "NA",
                ),
            )
            t_infer_start = time.time()
            try:
                state = self._pipe(**pipe_kwargs)
            except Exception as e:
                self._logger.exception("MiniMax-H3 推理调用失败")
                raise RuntimeError(f"MiniMax-H3 推理失败: {e}") from e
            finally:
                # v0.7.7+ T2.2：推理结束（成功或失败）重置 conditioning 缓存 key，
                # 防止残留被非 generate 路径的 TE 编码误用
                self._cond_cache_key = None
                # L2 模式：清理 text_encoder（如果 hook 未触发或推理失败）
                if _te_l2_moved_to_gpu:
                    try:
                        # 移除未触发的 hook
                        if _te_l2_hook_handle is not None:
                            _te_l2_hook_handle.remove()
                            _te_l2_hook_handle = None
                        # 确保 text_encoder 在 CPU（hook 可能已处理，这里兜底）
                        te = self._pipe.text_encoder
                        if hasattr(te, "device") and str(te.device).startswith("cuda"):
                            te.to("cpu")
                            torch.cuda.empty_cache()
                            self._logger.info("[L2 TE Cleanup] text_encoder 已移回 CPU（兜底）")
                    except Exception as _e:
                        self._logger.warning("[L2 TE Cleanup] text_encoder 清理失败: %s", _e)

                    # 确保被卸载的 vae 恢复到 GPU（hook 可能已处理，这里兜底）
                    for vae_name in _l2_vae_offloaded:
                        try:
                            vae_comp = getattr(self._pipe, vae_name, None)
                            if (vae_comp is not None and hasattr(vae_comp, "device")
                                    and not str(vae_comp.device).startswith("cuda")):
                                vae_comp.to(self._device)
                                self._logger.info(
                                    "[L2 VAE 恢复（兜底）] %s 已移回 GPU", vae_name)
                        except Exception as _e:
                            self._logger.warning(
                                "[L2 VAE 恢复（兜底）] %s 失败: %s", vae_name, _e)

                # L0 模式：推理后恢复 text_encoder 到 GPU
                if _te_offloaded:
                    try:
                        te = self._pipe.text_encoder
                        te.to(self._device)
                        self._logger.info("[L0 TE Offload] text_encoder 已恢复到 %s", self._device)
                    except Exception as _e:
                        self._logger.warning("[L0 TE Offload] text_encoder 恢复失败: %s", _e)

                # L3 模式：推理后把 transformer 和 vae 移回 CPU（释放显存）
                if _transformer_moved_to_gpu:
                    try:
                        # v0.6.12+ 修复：用 paradigm 判断（与移动到 GPU 时一致）
                        try:
                            _paradigm = (self._detect_paradigm(self._loaded_variant)
                                         if self._loaded_variant else "FL2VA")
                        except Exception:
                            _paradigm = "FL2VA"
                        transformer_attr = "transformer" if _paradigm == "FL2VA" else "transformer_ref"
                        tr = getattr(self._pipe, transformer_attr, None)
                        if tr is not None:
                            tr.to("cpu")
                            torch.cuda.empty_cache()
                            self._logger.info("[L3] %s 已移回 CPU（释放显存）", transformer_attr)
                    except Exception as _e:
                        self._logger.warning("[L3] transformer 移回 CPU 失败: %s", _e)

                if _vae_moved_to_gpu:
                    for vae_name in ("vae", "audio_vae"):
                        try:
                            vae_comp = getattr(self._pipe, vae_name, None)
                            if vae_comp is not None and hasattr(vae_comp, "device"):
                                if str(vae_comp.device).startswith("cuda"):
                                    vae_comp.to("cpu")
                                    torch.cuda.empty_cache()
                                    self._logger.info("[L3] %s 已移回 CPU", vae_name)
                        except Exception as _e:
                            self._logger.warning("[L3] %s 移回 CPU 失败: %s", vae_name, _e)

            infer_elapsed = time.time() - t_infer_start
            self._logger.info(
                "扩散推理完成: 耗时=%.2fs", infer_elapsed,
            )

            # ----- 提取视频 -----
            videos = state.get("videos")
            if videos is None or len(videos) == 0:
                raise RuntimeError("推理未生成视频（videos 为空）")

            video_array = videos[0]  # numpy 数组
            # 处理可能的 batch 维：np 输出形状 (1, T, H, W, C) 或 (T, H, W, C)
            if video_array.ndim == 5:
                video_array = video_array[0]  # (T, H, W, C)
            elif video_array.ndim == 4:
                pass  # 已是 (T, H, W, C)
            else:
                raise RuntimeError(
                    f"视频数组维度异常: ndim={video_array.ndim}, shape={video_array.shape}"
                )

            # 转换为 uint8 [0, 255]
            video_frames = (video_array * 255.0).clip(0, 255).astype(np.uint8)
            self._logger.info(
                "视频帧提取: shape=%s, dtype=%s",
                video_frames.shape, video_frames.dtype,
            )

            # ----- 提取音频 -----
            audio_data = None
            audio_state = state.get("audio")
            if audio_state is not None and len(audio_state) > 0:
                audio_array = audio_state[0]  # tensor (2, num_samples) 或类似
                # 转换为 numpy float32
                if hasattr(audio_array, "cpu"):
                    audio_array = audio_array.cpu().float().numpy()
                else:
                    audio_array = np.asarray(audio_array, dtype=np.float32)

                # MiniMax-H3 音频输出形状 (2, num_samples) 立体声
                # 兼容 (1, 2, num_samples) 等多余 batch 维
                while audio_array.ndim > 2:
                    audio_array = audio_array[0]

                audio_data = audio_array
                self._logger.info(
                    "音频提取: shape=%s, sr=%s",
                    audio_data.shape, state.get("sampling_rate"),
                )
            else:
                self._logger.info("推理未生成音频（mode=%s）", mode)

            elapsed = time.time() - start_time
            metadata = {
                "mode": mode,
                "variant": self._loaded_variant,
                "device": self._device,
                "offload_level": self._offload_level,
                "seed": seed,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance,
                "elapsed_seconds": elapsed,
                "infer_seconds": infer_elapsed,
                "is_placeholder": False,  # 真实推理
                "fps": fps,
                "duration": duration,
            }

            self._logger.info(
                "MiniMax-H3 生成完成: frames=%d, audio=%s, total=%.2fs, infer=%.2fs",
                video_frames.shape[0],
                "yes" if audio_data is not None else "no",
                elapsed, infer_elapsed,
            )

            return {
                "video_frames": video_frames,
                "audio_data": audio_data,
                "sampling_rate": state.get("sampling_rate", 32000),
                "metadata": metadata,
            }

    def _load_image(self, path_or_url: str):
        """加载图片为 PIL.Image 对象。

        Args:
            path_or_url: 本地文件路径或 http(s) URL

        Returns:
            PIL.Image.Image 对象（RGB）

        Raises:
            RuntimeError: 加载失败
        """
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                f"Pillow 未安装，无法加载图片: {e}"
            ) from e

        try:
            if path_or_url.startswith(("http://", "https://")):
                import io
                import urllib.request
                with urllib.request.urlopen(path_or_url, timeout=30) as resp:
                    data = resp.read()
                return Image.open(io.BytesIO(data)).convert("RGB")
            else:
                # 本地路径
                path = Path(path_or_url)
                if not path.exists():
                    raise FileNotFoundError(f"图片文件不存在: {path}")
                return Image.open(str(path)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"图片加载失败: {path_or_url}: {e}") from e

    def _load_video_frames(self, path_or_url: str, max_frames: int = 49):
        """加载视频为 PIL.Image 帧列表（用于 video_edit 模式）。

        使用 decord 解码视频，将每一帧转为 PIL.Image（RGB）。
        MiniMaxH3VideoReference.__init__ 接受 frames=list[PIL.Image]。

        Args:
            path_or_url: 本地视频文件路径或 http(s) URL
            max_frames: 最大帧数（防止显存爆炸，默认 49 帧≈2s@24fps）

        Returns:
            list[PIL.Image.Image]，RGB 模式

        Raises:
            RuntimeError: 加载失败
        """
        try:
            from PIL import Image
            import numpy as np
        except ImportError as e:
            raise RuntimeError(f"Pillow/numpy 未安装，无法加载视频: {e}") from e

        try:
            import decord
            decord.bridge.set_bridge("native")
        except ImportError as e:
            raise RuntimeError(
                f"decord 未安装，无法解码视频: {e}。请 pip install decord"
            ) from e

        try:
            self._logger.info("加载视频帧: %s (max_frames=%d)", path_or_url, max_frames)
            vr = decord.VideoReader(path_or_url)
            total = len(vr)
            fps = float(vr.get_avg_fps()) or 24.0
            # 抽帧：均匀采样，避免长视频占用过多显存
            if total > max_frames:
                indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()
            else:
                indices = list(range(total))
            frames = []
            for idx in indices:
                arr = vr[idx].asnumpy()  # (H, W, C) uint8
                img = Image.fromarray(arr, mode="RGB")
                frames.append(img)
            self._logger.info(
                "视频帧加载完成: total=%d, sampled=%d, fps=%.2f",
                total, len(frames), fps,
            )
            return frames, fps
        except Exception as e:
            raise RuntimeError(f"视频加载失败: {path_or_url}: {e}") from e

    def _cap_reference_vision_tokens(self, img, max_vision_tokens: int = 1024):
        """v0.7.7 方案3-b：封顶 ref2va/ref2v 参考图的视觉 token 数，降压长 token 编码。

        根因：Qwen3-VL conditioner 的 `image_processor.smart_resize` 对超高分辨率参考图
        会分片/放大 patch，单张参考图即可产生 ~7350 个图像 token（叠加音频/文本后总
        sequence 达 7792）。即便方案1 已把 TE attention 强制到 SDPA（线性显存），
        编码大量 token 仍会显著拉长时长并抬高峰值显存。这里在喂给管线前先把参考图
        等比缩放到「视觉 token 数 ≤ max_vision_tokens」的像素上限，从源头降压。

        换算依据（与 `encoders.py` 中 `num_image_tokens = int(grid.prod())//merge_size` 一致）：
        - patch_size=16、spatial_merge_size=2 → 单个合并视觉 token 覆盖 32×32 像素
        - 单 tile 参考图 token 数 ≈ 宽×高 / (32×32) = 像素数 / 1024
        - 故 max_vision_tokens=1024 ⇒ 参考图总像素 ≤ 1024×1024 ≈ 1048576

        Args:
            img: PIL.Image（RGB）
            max_vision_tokens: 单张参考图允许的最大视觉 token 数（默认 1024，
                空间细节与 token 量的折中，mean ~1024×1024 分辨率）

        Returns:
            PIL.Image：缩放后的参考图（若原图已足够小则原样返回）
        """
        try:
            from PIL import Image as _PILImage
        except Exception as _e:
            self._logger.warning("[Ref token cap] 无法导入 PIL，跳过参考图缩放: %s", _e)
            return img
        try:
            w, h = img.size
            cur_tokens = (w * h) // 1024  # 粗略估计当前视觉 token 数（单 tile）
            if cur_tokens <= max_vision_tokens:
                return img
            # 等比缩放到目标像素上限，同时保证宽高均为偶数（避免 patch 对齐问题）
            scale = min(1.0, (max_vision_tokens * 1024 / (w * h)) ** 0.5)
            nw = max(2, int((w * scale) // 2) * 2)
            nh = max(2, int((h * scale) // 2) * 2)
            resized = img.resize((nw, nh), _PILImage.LANCZOS)
            self._logger.info(
                "[Ref token cap] 参考图降压: %dx%d -> %dx%d "
                "（≈%d token -> ≈%d token，target<=%d）",
                w, h, nw, nh, cur_tokens, (nw * nh) // 1024, max_vision_tokens,
            )
            return resized
        except Exception as _e:
            self._logger.warning("[Ref token cap] 参考图缩放失败，使用原图: %s", _e)
            return img

    # ============ LoRA 管理 ============

    def load_lora(self, lora_path: str, weight: float = 1.0) -> dict:
        """加载单个 LoRA 到 transformer 模型（v0.7+ 真实实现）。

        复用 accelerators/lora_loader.py 的旁路 adapter 注入（与 T8/T4 turbo
        方案同一适配层）：ComfyUI 格式 key 自动映射到本地 ModularPipeline 模块，
        范式感知（transformer / transformer_ref 分区自动选择）。

        Args:
            lora_path: LoRA 文件路径（支持绝对路径 / 引擎目录相对 / 平台模型目录相对）
            weight: LoRA 权重（scale，1.0 = 原始强度）

        Returns:
            已加载的 LoRA 信息（含注入模块数）

        Raises:
            RuntimeError: 模型未加载 / LoRA 加载失败（0 个模块命中）
        """
        with self._lock:
            if self._pipe is None:
                raise RuntimeError("模型未加载，无法加载 LoRA")

            from .scheme_manager import LoRAConfig
            from .accelerators.lora_loader import apply_lora

            config = LoRAConfig(
                enabled=True,
                name=Path(lora_path).stem,
                path=lora_path,
                scale=weight,
            )
            ctx = apply_lora(self._pipe, config, logger=self._logger)
            if ctx is None:
                raise RuntimeError(
                    f"LoRA 注入失败（0 个模块命中，文件与模型结构不匹配）: {lora_path}"
                )

            name = Path(lora_path).stem
            # 同名重复挂载：先卸载旧的再挂新的（幂等）
            if name in self._manual_lora_ctxs:
                from .accelerators.lora_loader import restore_lora
                restore_lora(self._manual_lora_ctxs.pop(name), logger=self._logger)
            self._manual_lora_ctxs[name] = ctx

            return {
                "name": name,
                "path": str(lora_path),
                "weight": weight,
                "loaded": True,
                "injected_modules": ctx.injected,
                "skipped_modules": ctx.skipped,
            }

    def unload_lora(self, name: str) -> bool:
        """卸载单个 LoRA（v0.7+ 真实实现，恢复原 forward、移除旁路子模块）。

        Args:
            name: LoRA 名称

        Returns:
            是否成功卸载
        """
        with self._lock:
            ctx = self._manual_lora_ctxs.pop(name, None)
            if ctx is not None:
                from .accelerators.lora_loader import restore_lora
                restore_lora(ctx, logger=self._logger)
                return True
            # 非旁路挂载的 LoRA（如 LoRAManager 登记的元数据）
            if self._pipe is None:
                return self._lora_manager._unload_lora_unsafe(name)
            self._logger.warning("LoRA 卸载: 未找到已挂载的旁路 LoRA: name=%s", name)
            return False

    # ============ 卸载 ============

    def unload(self) -> None:
        """卸载所有模型组件，释放显存。"""
        with self._lock:
            self._unload_unsafe()

    def _unload_unsafe(self) -> None:
        """卸载模型（无锁版本）。

        v0.6+：在释放 pipeline 前先恢复所有加速方案的 monkey-patch，
        避免下次加载时残留 patch 影响新 pipeline 实例。
        """
        if self._pipe is None:
            return

        self._logger.info(
            "卸载 MiniMax-H3 模型: variant=%s, scheme=%s",
            self._loaded_variant, self._loaded_scheme_id,
        )

        # v0.6+ 恢复加速方案（必须先恢复再释放 pipeline）
        if self._acceleration_ctx is not None:
            try:
                from .accelerators import restore_acceleration
                restore_acceleration(self._acceleration_ctx, logger=self._logger)
            except Exception as e:
                self._logger.warning("加速方案恢复失败（忽略，继续释放 pipeline）: %s", e)
            self._acceleration_ctx = None

        # v0.7.2+ 恢复 SDNQ kernel_wrappers monkey-patch
        if hasattr(self, '_orig_int_scaled_mm_torch'):
            try:
                import sdnq.kernel_wrappers as _kw
                _kw.int_scaled_mm_torch = self._orig_int_scaled_mm_torch
                self._logger.debug("已恢复原始 int_scaled_mm_torch")
            except Exception as _e:
                self._logger.warning("恢复 int_scaled_mm_torch 失败（忽略）: %s", _e)
        if hasattr(self, '_orig_fp_mm_torch'):
            try:
                import sdnq.kernel_wrappers as _kw
                _kw.fp_mm_torch = self._orig_fp_mm_torch
                self._logger.debug("已恢复原始 fp_mm_torch")
            except Exception as _e:
                self._logger.warning("恢复 fp_mm_torch 失败（忽略）: %s", _e)

        # 释放 pipeline
        self._pipe = None
        self._loaded_variant = None
        self._loaded_scheme_id = None
        # v0.7.7+ T2.2：重置 conditioning 缓存 key（缓存文件保留，跨重启复用）
        self._cond_cache_key = None
        # v0.7+：手动挂载的 LoRA 旁路随 pipeline 一同释放，清空登记
        self._manual_lora_ctxs.clear()

        # v0.6.5+ 清空 SDNQ 组件缓存，释放系统 RAM
        # 原因：SDNQ 组件（text_encoder 21GB + transformer 18.8GB）缓存在 RAM 中，
        # 若不清空，切换 variant 时新加载的模型会与缓存叠加导致 OOM（58.6GB > 64GB RAM）。
        # 清空后切换 variant 需重新调用 sdnq.load_sdnq_model（无缓存复用），但避免 OOM。
        # 同一 variant 在不切换的情况下重载不受影响（unload+load 同一 variant 时缓存已清空，
        # 会重新加载，但不会 OOM 因为只有一个 variant 的模型在 RAM 中）。
        if self._cached_sdnq_components:
            cached_count = len(self._cached_sdnq_components)
            cached_keys = list(self._cached_sdnq_components.keys())
            self._cached_sdnq_components.clear()
            self._logger.info(
                "已清空 SDNQ 组件缓存: %d 个组件释放（keys=%s）",
                cached_count, cached_keys,
            )

        # 尝试释放显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self._logger.info("已清空 CUDA 缓存")
        except ImportError:
            pass
        except Exception as e:
            self._logger.warning("清理 CUDA 缓存失败: %s", e)

        # v0.6.5+ 强制 Python GC 释放已解除引用的 SDNQ 模型内存
        try:
            import gc
            gc.collect()
            self._logger.debug("已执行 gc.collect() 释放未引用内存")
        except Exception:
            pass

    # ============ 显存监控 ============

    def get_vram_used(self) -> float:
        """获取已用显存（MB）。

        Returns:
            已用显存 MB（无 GPU 或 torch 未安装返回 0.0）
        """
        try:
            import torch
            if torch.cuda.is_available() and "cuda" in self._device:
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        except Exception:
            pass
        return 0.0

    def _check_vram_available(self, device: str, required_gb: float) -> None:
        """显存预检：空闲显存不足时抛 RuntimeError。

        Args:
            device: 推理设备
            required_gb: 模型预估需要的显存（GB）

        Raises:
            RuntimeError: 显存不足（含友好提示）
        """
        if not device.startswith("cuda"):
            return  # CPU 模式无需预检

        try:
            import torch
            if not torch.cuda.is_available():
                return
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            if free_gb < required_gb:
                self._logger.warning(
                    "显存预检失败: free=%.2fGB < required=%.2fGB",
                    free_gb, required_gb,
                )
                raise RuntimeError(
                    f"GPU 显存不足：空闲 {free_gb:.2f}GB < 需要 {required_gb}GB。"
                    f"建议：1) 关闭其他占用显存的程序 2) 切换到更低的 offload 级别（L2 / L3）"
                    f" 3) 卸载其他已加载的引擎"
                )
            self._logger.info(
                "显存预检通过: free=%.2fGB >= required=%.2fGB",
                free_gb, required_gb,
            )
        except RuntimeError as e:
            if "显存不足" in str(e):
                raise  # 友好错误直接抛出
            self._logger.warning("显存预检异常（跳过）: %s", e)
        except Exception as e:
            self._logger.warning("显存预检异常（跳过）: %s", e)

    # ============ 工具方法 ============

    def _resolve_device(self, device: str) -> str:
        """解析设备字符串。

        Args:
            device: auto / cpu / cuda:0 / cuda:1

        Returns:
            实际设备字符串（auto → cuda:0 或 cpu）
        """
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda:0"
                return "cpu"
            except ImportError:
                return "cpu"
        return device

    @staticmethod
    def _parse_vram_required(vram_str: str) -> float:
        """解析显存需求字符串为数值（GB）。

        v0.6+：用于按方案显存需求做预检。

        Args:
            vram_str: 显存需求字符串（如 "24GB" / "48GB+" / "8GB"）

        Returns:
            显存数值（GB），解析失败返回 0.0（不做预检）
        """
        if not vram_str:
            return 0.0
        import re
        match = re.search(r"(\d+(?:\.\d+)?)", vram_str)
        if match:
            return float(match.group(1))
        return 0.0

    @staticmethod
    def _get_gpu_memory_used_mb() -> Optional[int]:
        """获取当前 GPU 已用显存（MB）。

        v0.6.9+：通过 nvidia-smi 查询，用于校验模型是否真正加载到 GPU。
        优先使用 nvidia-smi（跨进程，准确），fallback 到 torch.cuda（仅本进程）。

        Returns:
            已用显存（MB），无 GPU 或查询失败返回 None
        """
        import shutil
        import subprocess

        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                creation_flags = 0
                if sys.platform == "win32":
                    creation_flags = 0x08000000  # CREATE_NO_WINDOW
                result = subprocess.run(
                    [
                        nvidia_smi,
                        "--id=0",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=creation_flags,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return int(result.stdout.strip().split("\n")[0].strip())
            except Exception:
                pass

        # fallback：torch.cuda（仅本进程显存，不含其他进程）
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated() / 1024 / 1024)
        except Exception:
            pass

        return None
