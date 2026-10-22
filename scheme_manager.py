"""MiniMax-H3 多方案管理器（v2.0）。

加载与管理引擎的多个加载/推理方案（A/B/C/D），提供：
    - 方案配置加载（从 schemes/scheme_X.yaml）
    - 按显存自动选档（auto_select_scheme）
    - 方案合法性校验（适用 variant 检查）
    - 方案列表查询（供 UI / 平台 API 调用）

方案与 variant 的关系：
    - 方案 A/B/C：使用 SDNQ 4bit 量化版（OzzyGT/MiniMax_H3_sdnq_dynamic_4bit）
    - 方案 D：使用 BF16 原版（MiniMaxAI/MiniMax-H3）

详见 ANALYSIS-TASKS/MINIMAX_H3_MULTI_SCHEME_ANALYSIS.md §5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ===== 默认方案选择阈值（GB）=====
# 与 config.py _OFFLOAD_VRAM_THRESHOLDS 保持一致
# 按显存自动选档策略：
#   ≥48GB → 方案 D（BF16 原版，最高质量）
#   ≥24GB → 方案 A（SDNQ 4bit，24G 流畅版）
#   ≥12GB → 方案 B（SDNQ 4bit，12-16G 可用版）
#   <12GB → 方案 C（SDNQ 4bit，8G 极低配版）
_AUTO_SELECT_THRESHOLDS = {
    "D": 48.0,
    "A": 24.0,
    "B": 12.0,
    # 其余走 C
}


@dataclass
class SageAttentionConfig:
    """SageAttention 加速方案配置。"""
    enabled: bool = False
    backend: str = "int8_fp8"  # INT8 QK + FP8 PV


@dataclass
class LoRAConfig:
    """Turbo LoRA 加速方案配置（v0.7+ 适配官方 Comfy-Org turbo LoRA）。

    v0.7 新增 ref2v_path：Ref2VA 范式专用 LoRA 路径（官方 ref2v turbo 与
    fl2v turbo 是两个文件，方案按当前范式自动选择）。
    """
    enabled: bool = False
    name: str = ""
    path: str = ""
    ref2v_path: str = ""   # v0.7+ Ref2VA 范式专用 LoRA（空则 Ref2VA 也用 path）
    scale: float = 1.0
    supports_paradigms: list[str] = field(default_factory=list)


@dataclass
class FirstBlockCacheConfig:
    """FirstBlockCache 加速方案配置。"""
    enabled: bool = False
    threshold: float = 0.05  # 残差阈值，小于此值复用上次结果


@dataclass
class TeSpeedCacheConfig:
    """TE-Speed 块级缓存配置（v0.7.7+ T3.1，移植原理）。

    语义（对应官方 TE-Speed-MiniMaxH3-OSS 参数）：
        - control_value：sigma 差阈值（默认 0.12）；相邻去噪 step 的 sigma 差小于
          此值且处于缓存窗口时，尾部块整段跳过（复用上一步段输出）
        - percent_1：缓存窗口起点比例（默认 0.1，前 10% step 强制完全计算）
        - percent_2：缓存窗口终点比例（默认 0.9，最后 10% step 强制完全计算）
        - start_block_ratio：尾部段起点（默认 0.5，即后 50% 块参与缓存；
          官方最大约 75%，显存受限时调低）
        - device：缓存存储设备（"cuda" 速度优先；"cpu" 命中时搬回 GPU 一次，省显存）
    """
    enabled: bool = False
    control_value: float = 0.12
    percent_1: float = 0.1
    percent_2: float = 0.9
    start_block_ratio: float = 0.5
    device: str = "cuda"


@dataclass
class AccelerationConfig:
    """加速方案总配置。"""
    sage_attention: SageAttentionConfig = field(default_factory=SageAttentionConfig)
    lora: Optional[LoRAConfig] = None
    first_block_cache: Optional[FirstBlockCacheConfig] = None
    te_speed_cache: Optional[TeSpeedCacheConfig] = None  # v0.7.7+ T3.1


@dataclass
class BlockSwapConfig:
    """BlockSwap 显存优化配置（方案 C 用）。"""
    enabled: bool = False
    swap_blocks: int = 20       # 把 N 个 transformer 块卸载到 CPU
    swap_strategy: str = "even"  # 均匀分布


@dataclass
class VAETilingConfig:
    """VAE tiling 显存优化配置（方案 C 用）。"""
    enabled: bool = False
    tile_size: int = 256


@dataclass
class InferenceConfig:
    """推理参数配置。"""
    num_inference_steps: int = 30
    scheduler: str = "simple"
    sampler: str = "euler"
    guidance_scale: float = 7.0
    default_resolution: str = "768x432"
    default_fps: int = 24
    default_duration: int = 5
    # v1.1+ sigma shift（对齐官方 turbo 蒸馏参数）
    # diffusers MiniMaxH3Scheduler 固定 Euler + 指数 shift，shift 可每请求覆盖：
    #   官方 8 步版  shift=12/3（视频/音频）
    #   官方 4 步版  shift=6/3（视频/音频）
    flow_shift: float = 12.0        # 视频调度 shift
    audio_flow_shift: float = 3.0   # 音频调度 shift


@dataclass
class SchemeConfig:
    """单方案配置（对应 schemes/scheme_X.yaml）。

    Attributes:
        scheme: 方案 ID（A / B / C / D）
        name: 方案中文名
        description: 方案描述
        vram_required: 显存需求标签（如 "24GB" / "48GB+"）
        model_repo: 模型 HuggingFace 仓库 ID
        model_subdir: 模型本地子目录（相对 models_root）
        precision: 精度标签（sdnq_uint4 / sdnq_uint4_fp16 / bf16）
        size_gb: 模型大小（GB，用于下载提示）
        offload_level: CPU offload 级别（L0 / L1 / L2 / L3）
        device: 推理设备（cuda:0 / cpu）
        acceleration: 加速方案配置
        block_swap: BlockSwap 配置
        vae_tiling: VAE tiling 配置
        inference: 推理参数
        applicable_variants: 适用 variant 名称列表
    """
    scheme: str
    name: str
    description: str
    vram_required: str
    model_repo: str
    model_subdir: str
    precision: str
    size_gb: int
    offload_level: str
    device: str = "cuda:0"
    acceleration: AccelerationConfig = field(default_factory=AccelerationConfig)
    block_swap: BlockSwapConfig = field(default_factory=BlockSwapConfig)
    vae_tiling: VAETilingConfig = field(default_factory=VAETilingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    applicable_variants: list[str] = field(default_factory=list)


class SchemeManager:
    """方案管理器。

    负责：
        - 从 schemes/ 目录加载所有方案 yaml
        - 按显存自动选档
        - 方案合法性校验
        - 提供方案列表给 UI / 平台 API

    用法：
        manager = SchemeManager(schemes_dir=Path("engines/minimax_h3_video_a1b2c3/schemes"))
        scheme = manager.auto_select_scheme(vram_gb=24.0)
        all_schemes = manager.list_schemes()
    """

    def __init__(
        self,
        schemes_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化方案管理器。

        Args:
            schemes_dir: schemes/ 目录路径（包含 scheme_X.yaml）
            logger: 可选的日志器
        """
        self.schemes_dir = schemes_dir
        self._logger = logger or logging.getLogger(__name__)
        self._cache: dict[str, SchemeConfig] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """懒加载所有方案 yaml 到缓存。"""
        if self._loaded:
            return

        if not self.schemes_dir.exists():
            self._logger.warning("schemes 目录不存在: %s", self.schemes_dir)
            self._loaded = True
            return

        for yaml_file in sorted(self.schemes_dir.glob("scheme_*.yaml")):
            try:
                scheme = self._load_yaml(yaml_file)
                if scheme is not None:
                    self._cache[scheme.scheme] = scheme
                    self._logger.info(
                        "加载方案 %s: %s（vram_required=%s, precision=%s）",
                        scheme.scheme, scheme.name, scheme.vram_required, scheme.precision,
                    )
            except Exception as e:
                self._logger.exception("加载方案文件失败: %s (%s)", yaml_file, e)

        self._loaded = True

        if not self._cache:
            self._logger.warning("未在 %s 中找到任何方案文件", self.schemes_dir)

    def _load_yaml(self, yaml_path: Path) -> Optional[SchemeConfig]:
        """解析单个 scheme_X.yaml 为 SchemeConfig。

        Args:
            yaml_path: yaml 文件路径

        Returns:
            SchemeConfig 实例，解析失败返回 None
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        # 解析 acceleration 块
        accel_data = data.get("acceleration", {}) or {}
        sage_data = accel_data.get("sage_attention", {}) or {}
        lora_data = accel_data.get("lora", None)
        fbc_data = accel_data.get("first_block_cache", None)
        te_data = accel_data.get("te_speed_cache", None)

        acceleration = AccelerationConfig(
            sage_attention=SageAttentionConfig(
                enabled=bool(sage_data.get("enabled", False)),
                backend=str(sage_data.get("backend", "int8_fp8")),
            ),
            lora=LoRAConfig(
                enabled=bool(lora_data.get("enabled", False)) if lora_data else False,
                name=str(lora_data.get("name", "")) if lora_data else "",
                path=str(lora_data.get("path", "")) if lora_data else "",
                ref2v_path=str(lora_data.get("ref2v_path", "")) if lora_data else "",
                scale=float(lora_data.get("scale", 1.0)) if lora_data else 1.0,
                supports_paradigms=list(lora_data.get("supports_paradigms", [])) if lora_data else [],
            ) if lora_data else None,
            first_block_cache=FirstBlockCacheConfig(
                enabled=bool(fbc_data.get("enabled", False)) if fbc_data else False,
                threshold=float(fbc_data.get("threshold", 0.05)) if fbc_data else 0.05,
            ) if fbc_data else None,
            te_speed_cache=TeSpeedCacheConfig(
                enabled=bool(te_data.get("enabled", False)) if te_data else False,
                control_value=float(te_data.get("control_value", 0.12)) if te_data else 0.12,
                percent_1=float(te_data.get("percent_1", 0.1)) if te_data else 0.1,
                percent_2=float(te_data.get("percent_2", 0.9)) if te_data else 0.9,
                start_block_ratio=float(te_data.get("start_block_ratio", 0.5)) if te_data else 0.5,
                device=str(te_data.get("device", "cuda")) if te_data else "cuda",
            ) if te_data else None,
        )

        # 解析 block_swap 块
        bs_data = data.get("block_swap", {}) or {}
        block_swap = BlockSwapConfig(
            enabled=bool(bs_data.get("enabled", False)),
            swap_blocks=int(bs_data.get("swap_blocks", 20)),
            swap_strategy=str(bs_data.get("swap_strategy", "even")),
        )

        # 解析 vae_tiling 块
        vt_data = data.get("vae_tiling", {}) or {}
        vae_tiling = VAETilingConfig(
            enabled=bool(vt_data.get("enabled", False)),
            tile_size=int(vt_data.get("tile_size", 256)),
        )

        # 解析 inference 块
        inf_data = data.get("inference", {}) or {}
        inference = InferenceConfig(
            num_inference_steps=int(inf_data.get("num_inference_steps", 30)),
            scheduler=str(inf_data.get("scheduler", "simple")),
            sampler=str(inf_data.get("sampler", "euler")),
            guidance_scale=float(inf_data.get("guidance_scale", 7.0)),
            default_resolution=str(inf_data.get("default_resolution", "768x432")),
            default_fps=int(inf_data.get("default_fps", 24)),
            default_duration=int(inf_data.get("default_duration", 5)),
            flow_shift=float(inf_data.get("flow_shift", 12.0)),
            audio_flow_shift=float(inf_data.get("audio_flow_shift", 3.0)),
        )

        return SchemeConfig(
            scheme=str(data["scheme"]),
            name=str(data["name"]),
            description=str(data["description"]),
            vram_required=str(data["vram_required"]),
            model_repo=str(data["model_repo"]),
            model_subdir=str(data["model_subdir"]),
            precision=str(data["precision"]),
            size_gb=int(data["size_gb"]),
            offload_level=str(data["offload_level"]),
            device=str(data.get("device", "cuda:0")),
            acceleration=acceleration,
            block_swap=block_swap,
            vae_tiling=vae_tiling,
            inference=inference,
            applicable_variants=list(data.get("applicable_variants", [])),
        )

    def load_scheme(self, scheme_id: str) -> Optional[SchemeConfig]:
        """按 ID 加载单个方案。

        Args:
            scheme_id: 方案 ID（A / B / C / D）

        Returns:
            SchemeConfig，不存在返回 None
        """
        self._ensure_loaded()
        return self._cache.get(scheme_id)

    def list_schemes(self) -> list[SchemeConfig]:
        """列出所有方案（按 ID 字母序）。"""
        self._ensure_loaded()
        return [self._cache[k] for k in sorted(self._cache.keys())]

    def auto_select_scheme(self, vram_gb: Optional[float] = None) -> Optional[SchemeConfig]:
        """按 GPU 显存自动选择方案。

        策略：
            ≥48GB → 方案 D（BF16 原版）
            ≥24GB → 方案 A（SDNQ 4bit，24G 流畅版）
            ≥12GB → 方案 B（SDNQ 4bit，12-16G 可用版）
            <12GB → 方案 C（SDNQ 4bit，8G 极低配版）

        Args:
            vram_gb: GPU 显存（GB），None 时自动检测

        Returns:
            推荐的 SchemeConfig，无可用方案返回 None
        """
        self._ensure_loaded()

        if vram_gb is None:
            vram_gb = _detect_gpu_vram_gb()

        self._logger.info("自动选档: 检测到 GPU 显存=%.1fGB", vram_gb)

        # 按阈值降序选择
        for scheme_id in ["D", "A", "B", "C"]:
            threshold = _AUTO_SELECT_THRESHOLDS.get(scheme_id, 0.0)
            if vram_gb >= threshold and scheme_id in self._cache:
                selected = self._cache[scheme_id]
                self._logger.info(
                    "自动选档结果: 方案 %s（%s），阈值=%.0fGB，实际=%.1fGB",
                    selected.scheme, selected.name, threshold, vram_gb,
                )
                return selected

        # 都不满足，返回最低档（C）或第一个可用方案
        fallback = self._cache.get("C") or (
            self.list_schemes()[0] if self.list_schemes() else None
        )
        if fallback:
            self._logger.warning(
                "显存 %.1fGB 不满足任何方案阈值，降级使用方案 %s",
                vram_gb, fallback.scheme,
            )
        return fallback

    def validate_scheme_for_variant(
        self,
        scheme: SchemeConfig,
        variant_name: str,
    ) -> bool:
        """校验方案是否适用于指定 variant。

        方案与 variant 必须匹配：
            - 方案 A/B/C 适用于 SDNQ 4bit variant（如 MiniMax/H3-FL2VA-SDNQ）
            - 方案 D 适用于 BF16 原版 variant（如 MiniMax/H3-FL2VA）

        Args:
            scheme: 方案配置
            variant_name: variant 名称

        Returns:
            是否适用
        """
        if not scheme.applicable_variants:
            # 未声明 applicable_variants 时不做限制
            return True

        applicable = variant_name in scheme.applicable_variants
        if not applicable:
            self._logger.warning(
                "方案 %s 不适用于 variant %s（适用列表: %s）",
                scheme.scheme, variant_name, scheme.applicable_variants,
            )
        return applicable

    def get_scheme_for_variant(self, variant_name: str) -> Optional[SchemeConfig]:
        """根据 variant 名称反查适用的方案。

        优先级：
            1. 如果 variant 名带 -SDNQ 后缀，返回方案 A
            2. 如果 variant 名是 BF16 原版，返回方案 D
            3. 否则按显存自动选档

        Args:
            variant_name: variant 名称

        Returns:
            推荐的 SchemeConfig
        """
        self._ensure_loaded()

        if "SDNQ" in variant_name.upper() or "4bit" in variant_name.lower():
            # SDNQ 4bit variant → 默认方案 A
            return self._cache.get("A") or self.auto_select_scheme()
        else:
            # BF16 原版 → 方案 D
            return self._cache.get("D") or self.auto_select_scheme()


def _detect_gpu_vram_gb() -> float:
    """检测当前 GPU 显存（GB），无 GPU 或检测失败返回 0.0。

    与 config.py 的 _detect_gpu_vram_gb 实现一致，避免循环依赖。
    """
    # 优先用 torch
    try:
        import torch
        if torch.cuda.is_available():
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return total_bytes / (1024 ** 3)
    except Exception:
        pass

    # fallback：nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().splitlines()[0].strip()) / 1024
    except Exception:
        pass

    return 0.0
