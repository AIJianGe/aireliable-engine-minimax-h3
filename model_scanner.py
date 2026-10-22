"""MiniMax-H3 模型文件扫描器（多方案版 v0.6+）。

扫描 models/video/minimax-h3/ 目录，识别可用的 MiniMax-H3 模型变体。

v0.6+ 多方案架构升级（v2.0）：
    - 双套模型文件：SDNQ 4bit（sdnq_format/）+ BF16 原版（diffusers_format/）
    - 4 个变体：SDNQ FL2VA / SDNQ Ref2VA / BF16 FL2VA / BF16 Ref2VA
    - 每个变体声明适用的方案（schemes 字段）

v0.2.11+ 方案 B 改造：
    优先调用平台 API（`GET /engines/{id}/variants`）获取 variant 可用性，
    平台不可达时 fallback 到本地文件扫描（检查标志文件存在性）。

本地 fallback 检查策略（遵循 project-rules.md 第二部分 §5）：
    - 只检查标志文件（config.json 存在性），不严格校验大小
    - 让 variant 在 UI 中可见（可触发下载），不阻断启动
    - 完整性校验由平台 API（key_files 机制）或 wrapper.load() 兜底

模型目录结构（双套并存，与 manifest.yaml 一致）：
    models/video/minimax-h3/
    ├── sdnq_format/                     ← SDNQ 4bit 量化版（方案 A/B/C 用）
    │   ├── model_index.json
    │   ├── modular_model_index.json
    │   ├── transformer/                 ← FL2VA transformer（量化版）
    │   │   └── config.json
    │   ├── transformer_ref/             ← Ref2VA transformer（量化版）
    │   │   └── config.json
    │   ├── text_encoder/                 ← Qwen3VL-32B 文本编码器（量化版）
    │   ├── vae/                          ← 视频 VAE
    │   ├── audio_vae/                    ← 音频 VAE
    │   ├── scheduler/                    ← 视频调度器
    │   ├── audio_scheduler/              ← 音频调度器
    │   ├── processor/                    ← Qwen3VL 处理器
    │   ├── tokenizer/                    ← Qwen2 分词器
    │   └── loras/                        ← 加速 LoRA（LightX2V Turbo）
    └── diffusers_format/                 ← BF16 原版（方案 D 用）
        ├── model_index.json
        ├── modular_model_index.json
        ├── transformer/                  ← FL2VA transformer（BF16 原版）
        │   └── config.json
        ├── transformer_ref/             ← Ref2VA transformer（BF16 原版）
        │   └── config.json
        ├── text_encoder/                 ← Qwen3VL-32B 文本编码器（BF16 原版）
        └── ...（其他共享组件）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .schemas import VideoVariantInfo

logger = logging.getLogger(__name__)


# ===== 已知模型变体定义（v0.6+ 多方案版，4 个变体）=====
# v0.6+ 双套模型文件：
#   - SDNQ 4bit 量化版（方案 A/B/C 用）：sdnq_format/ 子目录
#   - BF16 原版（方案 D 用）：diffusers_format/ 子目录
#
# subdir 字段为相对于 model_dir（即 video/minimax-h3/）的子目录。
# v0.6+ model_dir 是 video/minimax-h3/ 根目录（不再是 diffusers_format/），
# 所以 subdir 需要包含 sdnq_format/ 或 diffusers_format/ 前缀。
_KNOWN_VARIANTS: list[dict] = [
    # ============================================================
    # 1. SDNQ 4bit FL2VA（方案 A/B/C 用，推荐默认）
    # ============================================================
    {
        "name": "MiniMax/H3-FL2VA-SDNQ",
        "alias": "H3-FL2VA-4bit",
        "base_model": "FL2VA",
        "precision": "sdnq_uint4",
        "loader": "minimax",
        "vram_required": "24GB",  # 方案 A 最低显存需求
        "subdir": "sdnq_format/transformer",  # 相对 model_dir（video/minimax-h3/）
        "resolution": "768x432",
        "marker_files": ["config.json"],
        "required_files": [
            "sdnq_format/transformer/config.json",
            "sdnq_format/text_encoder/config.json",
            "sdnq_format/vae/config.json",
            "sdnq_format/audio_vae/config.json",
            "sdnq_format/scheduler/scheduler_config.json",
            "sdnq_format/audio_scheduler/scheduler_config.json",
            "sdnq_format/model_index.json",
            "sdnq_format/modular_model_index.json",
        ],
        "supports_audio": True,
        "schemes": ["A", "B", "C"],  # v0.6+ 新增：适用的方案
    },
    # ============================================================
    # 2. SDNQ 4bit Ref2VA（方案 A/B/C 用）
    # ============================================================
    {
        "name": "MiniMax/H3-Ref2VA-SDNQ",
        "alias": "H3-Ref2VA-4bit",
        "base_model": "Ref2VA",
        "precision": "sdnq_uint4",
        "loader": "minimax",
        "vram_required": "24GB",
        "subdir": "sdnq_format/transformer_ref",
        "resolution": "1280x720",
        "marker_files": ["config.json"],
        "required_files": [
            "sdnq_format/transformer_ref/config.json",
            "sdnq_format/text_encoder/config.json",
            "sdnq_format/vae/config.json",
            "sdnq_format/audio_vae/config.json",
            "sdnq_format/scheduler/scheduler_config.json",
            "sdnq_format/audio_scheduler/scheduler_config.json",
            "sdnq_format/model_index.json",
            "sdnq_format/modular_model_index.json",
        ],
        "supports_audio": True,
        "schemes": ["A", "B", "C"],
    },
    # ============================================================
    # 3. BF16 原版 FL2VA（方案 D 用，最高质量）
    # ============================================================
    {
        "name": "MiniMax/H3-FL2VA",
        "alias": "H3-FL2VA-BF16",
        "base_model": "FL2VA",
        "precision": "bf16",
        "loader": "minimax",
        "vram_required": "48GB+",  # BF16 原版需 80GB+ 显存才能流畅运行
        "subdir": "diffusers_format/transformer",
        "resolution": "768x432",
        "marker_files": ["config.json"],
        "required_files": [
            "diffusers_format/transformer/config.json",
            "diffusers_format/text_encoder/config.json",
            "diffusers_format/vae/config.json",
            "diffusers_format/audio_vae/config.json",
            "diffusers_format/scheduler/scheduler_config.json",
            "diffusers_format/audio_scheduler/scheduler_config.json",
            "diffusers_format/model_index.json",
            "diffusers_format/modular_model_index.json",
        ],
        "supports_audio": True,
        "schemes": ["D"],  # v0.6+ BF16 原版仅适用方案 D
    },
    # ============================================================
    # 4. BF16 原版 Ref2VA（方案 D 用）
    # ============================================================
    {
        "name": "MiniMax/H3-Ref2VA",
        "alias": "H3-Ref2VA-BF16",
        "base_model": "Ref2VA",
        "precision": "bf16",
        "loader": "minimax",
        "vram_required": "48GB+",
        "subdir": "diffusers_format/transformer_ref",
        "resolution": "1280x720",
        "marker_files": ["config.json"],
        "required_files": [
            "diffusers_format/transformer_ref/config.json",
            "diffusers_format/text_encoder/config.json",
            "diffusers_format/vae/config.json",
            "diffusers_format/audio_vae/config.json",
            "diffusers_format/scheduler/scheduler_config.json",
            "diffusers_format/audio_scheduler/scheduler_config.json",
            "diffusers_format/model_index.json",
            "diffusers_format/modular_model_index.json",
        ],
        "supports_audio": True,
        "schemes": ["D"],
    },
]


class ModelScanner:
    """模型目录扫描器（多方案版 v0.6+）。

    扫描 model_dir/ 下的 sdnq_format/ 与 diffusers_format/ 子目录，
    识别已下载的 MiniMax-H3 模型变体（SDNQ 4bit + BF16 双套）。

    v0.2.11+ 方案 B：优先调用平台 API，fallback 到本地扫描。
    """

    def __init__(
        self,
        models_root: Path,
        model_dir: Path,
        engine=None,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化扫描器。

        Args:
            models_root: 模型根目录（如 models/），用于 LoRA 扫描相对路径解析
            model_dir: MiniMax-H3 模型目录
                v0.6+ 为 video/minimax-h3/（根目录，包含 sdnq_format/ 与 diffusers_format/ 子目录）
                v0.5 及之前为 video/minimax-h3/diffusers_format/（仅 BF16 一套）
            engine: 可选的 BaseEngine 实例（v0.2.11+ 方案 B 注入）。
                传入时优先调用平台 API 检测 variant；
                为 None 时走 fallback 本地扫描逻辑。
            logger: 可选的日志器
        """
        self.models_root = models_root
        self.model_dir = model_dir
        self.engine = engine
        self._logger = logger or logging.getLogger(__name__)

    async def scan_variants_async(self) -> list[VideoVariantInfo]:
        """v0.2.11+ 方案 B：异步扫描，优先调用平台 API。

        调用链：
            1. engine.get_variants() → 平台 API（声明式检测）
            2. 平台返回 None → fallback 到 _scan_locally()
            3. 合并平台 available 字段 + _KNOWN_VARIANTS 引擎特有字段
        """
        if self.engine is not None:
            platform_variants = await self.engine.get_variants()
            if platform_variants is not None:
                return self._merge_with_known(platform_variants)

        # fallback：本地扫描
        return self._scan_locally()

    def scan_variants(self) -> list[VideoVariantInfo]:
        """同步扫描接口（向后兼容）。

        v0.2.11+ 注：若需使用方案 B 平台 API，请调用 scan_variants_async()。
        此同步方法走 fallback 本地扫描逻辑。
        """
        return self._scan_locally()

    def _merge_with_known(self, platform_variants: list[dict]) -> list[VideoVariantInfo]:
        """合并平台 variant 数据 + _KNOWN_VARIANTS 引擎特有字段。

        Args:
            platform_variants: 平台 API 返回的 variant 列表（含 available 字段）

        Returns:
            VideoVariantInfo 列表（available 来自平台，supports_audio / schemes 等来自 _KNOWN_VARIANTS）
        """
        known_lookup: dict[str, dict] = {v["name"]: v for v in _KNOWN_VARIANTS}

        result: list[VideoVariantInfo] = []
        for pv in platform_variants:
            v_name = pv.get("name", "")
            kv = known_lookup.get(v_name, {})
            available = bool(pv.get("available", False))

            result.append(VideoVariantInfo(
                name=v_name,
                base_model=pv.get("base_model", kv.get("base_model", "")),
                precision=pv.get("precision", kv.get("precision", "bf16")),
                loader=pv.get("loader", kv.get("loader", "minimax")),
                vram_required=pv.get("vram_required", kv.get("vram_required", "未知")),
                available=available,
                resolution=pv.get("resolution", kv.get("resolution", "768x432")),
                fps=24,
                duration=5,
                max_duration=15,
                supports_audio=kv.get("supports_audio", True),
            ))
            if available:
                self._logger.info("发现模型变体: %s (%s)", v_name, pv.get("alias", ""))

        if not any(pv.get("available") for pv in platform_variants):
            self._logger.warning(
                "未在 %s 中找到任何可用的 MiniMax-H3 模型变体，"
                "请到模型管理界面下载（SDNQ 仓库：OzzyGT/MiniMax_H3_sdnq_dynamic_4bit，"
                "BF16 仓库：MiniMaxAI/MiniMax-H3）",
                self.model_dir,
            )

        return result

    def _scan_locally(self) -> list[VideoVariantInfo]:
        """fallback 本地扫描（原 scan_variants 逻辑）。

        本地 fallback 策略（遵循 project-rules.md 第二部分 §5）：
            - 只检查标志文件（config.json）存在性
            - 不严格校验大小（避免断点续传残文件被误判可用）
            - 让 variant 在 UI 中可见，触发下载入口
        """
        capabilities: list[VideoVariantInfo] = []

        if not self.model_dir.exists():
            self._logger.warning("模型目录不存在: %s", self.model_dir)
            return capabilities

        # v0.6+ 检查双套目录
        sdnq_dir = self.model_dir / "sdnq_format"
        bf16_dir = self.model_dir / "diffusers_format"
        if not sdnq_dir.exists() and not bf16_dir.exists():
            self._logger.warning(
                "未找到 SDNQ 4bit 目录（%s）或 BF16 原版目录（%s），"
                "请到模型管理界面下载",
                sdnq_dir, bf16_dir,
            )
            return capabilities

        for variant_def in _KNOWN_VARIANTS:
            variant_dir = self.model_dir / variant_def["subdir"]

            if not variant_dir.exists():
                self._logger.debug(
                    "变体 %s 目录不存在: %s", variant_def["name"], variant_dir,
                )
                continue

            # 本地 fallback 只检查标志文件（marker_files）
            has_model = self._check_variant_files(variant_def["name"])
            if not has_model:
                self._logger.debug(
                    "变体 %s 标志文件缺失: %s", variant_def["name"], variant_dir,
                )
                continue

            cap = VideoVariantInfo(
                name=variant_def["name"],
                base_model=variant_def["base_model"],
                precision=variant_def["precision"],
                loader=variant_def["loader"],
                vram_required=variant_def["vram_required"],
                available=True,
                resolution=variant_def.get("resolution", "768x432"),
                fps=24,
                duration=5,
                max_duration=15,
                supports_audio=variant_def.get("supports_audio", True),
            )
            capabilities.append(cap)
            self._logger.info(
                "发现模型变体: %s (%s, schemes=%s)",
                cap.name, variant_def["alias"], variant_def.get("schemes", []),
            )

        if not capabilities:
            self._logger.warning(
                "未在 %s 中找到任何 MiniMax-H3 模型变体，"
                "请到模型管理界面下载（SDNQ 仓库：OzzyGT/MiniMax_H3_sdnq_dynamic_4bit，"
                "BF16 仓库：MiniMaxAI/MiniMax-H3）",
                self.model_dir,
            )

        return capabilities

    def _check_variant_files(self, variant_name: str) -> bool:
        """检查 variant 所需的标志文件是否存在。

        本地 fallback 策略：只检查 marker_files（transformer config.json），
        不严格校验共享组件（text_encoder / vae）的大小。
        完整性校验由平台 API 或 wrapper.load() 兜底。

        Args:
            variant_name: variant 名称（如 MiniMax/H3-FL2VA-SDNQ）

        Returns:
            标志文件是否齐全
        """
        variant_def = None
        for v in _KNOWN_VARIANTS:
            if v["name"] == variant_name:
                variant_def = v
                break
        if variant_def is None:
            return False

        variant_dir = self.model_dir / variant_def["subdir"]
        for marker in variant_def.get("marker_files", []):
            if not (variant_dir / marker).exists():
                return False
        return True

    def resolve_variant_path(self, variant_name: str) -> Optional[Path]:
        """根据变体名解析 transformer 模型路径。

        Args:
            variant_name: 变体名（如 MiniMax/H3-FL2VA-SDNQ）

        Returns:
            transformer 模型目录路径（如 models/video/minimax-h3/sdnq_format/transformer），
            不存在返回 None
        """
        for variant_def in _KNOWN_VARIANTS:
            if variant_def["name"] == variant_name:
                path = self.model_dir / variant_def["subdir"]
                return path if path.exists() else None
        return None

    def get_variant_definition(self, variant_name: str) -> Optional[dict]:
        """获取 variant 的完整定义（含 required_files 清单 + schemes）。

        供 wrapper.load() 用于文件完整性校验，以及 SchemeManager 用于方案匹配。

        Args:
            variant_name: variant 名称

        Returns:
            variant 定义字典（含 schemes 字段），不存在返回 None
        """
        for v in _KNOWN_VARIANTS:
            if v["name"] == variant_name:
                return v
        return None

    def get_variants_by_scheme(self, scheme_id: str) -> list[dict]:
        """按方案 ID 查询适用的 variant 定义列表。

        v0.6+ 新增：方案与 variant 双向关联查询。

        Args:
            scheme_id: 方案 ID（A / B / C / D）

        Returns:
            适用的 variant 定义列表（_KNOWN_VARIANTS 中的字典）
        """
        return [
            v for v in _KNOWN_VARIANTS
            if scheme_id in v.get("schemes", [])
        ]

    def scan_loras(self) -> list[str]:
        """扫描 LoRA 目录下的 .safetensors 文件名。

        v0.6+ 双 LoRA 目录：
            - sdnq_format/loras/：加速 LoRA（LightX2V Turbo，方案 A/B/C 自动加载）
            - diffusers_format/loras/：用户 LoRA（运行时动态切换）

        本方法扫描两个目录并合并结果。

        Returns:
            LoRA 名称列表（文件 stem，不含扩展名）
        """
        names: list[str] = []

        # 1. 加速 LoRA 目录（sdnq_format/loras/）
        sdnq_lora_dir = self.model_dir / "sdnq_format" / "loras"
        if sdnq_lora_dir.exists():
            sdnq_names = sorted(f.stem for f in sdnq_lora_dir.glob("*.safetensors"))
            names.extend(sdnq_names)
            self._logger.info(
                "加速 LoRA 扫描（sdnq_format）: 找到 %d 个 (dir=%s)",
                len(sdnq_names), sdnq_lora_dir,
            )

        # 2. 用户 LoRA 目录（diffusers_format/loras/）
        bf16_lora_dir = self.model_dir / "diffusers_format" / "loras"
        if bf16_lora_dir.exists():
            bf16_names = sorted(f.stem for f in bf16_lora_dir.glob("*.safetensors"))
            # 去重（避免与 SDNQ 目录同名文件冲突）
            for n in bf16_names:
                if n not in names:
                    names.append(n)
            self._logger.info(
                "用户 LoRA 扫描（diffusers_format）: 找到 %d 个 (dir=%s)",
                len(bf16_names), bf16_lora_dir,
            )

        if not names:
            self._logger.debug("LoRA 目录不存在或为空（已检查 sdnq_format/loras 和 diffusers_format/loras）")

        return names
