"""方案合法性校验器（v2.0）。

在切换方案前校验：
    - 方案与 variant 是否匹配（SDNQ variant 不能用方案 D，BF16 variant 不能用方案 A/B/C）
    - 显存是否满足方案要求
    - 模型文件是否完整（按 variant 的 model_file_ids 检查）

返回 ValidationResult，含通过/失败 + 失败原因列表。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .scheme_manager import SchemeConfig, SchemeManager


@dataclass
class ValidationResult:
    """方案校验结果。"""
    ok: bool
    scheme_id: str
    variant_name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SchemeValidator:
    """方案校验器。

    用法：
        validator = SchemeValidator(scheme_manager, models_root=Path("models"))
        result = validator.validate(scheme_id="A", variant_name="MiniMax/H3-FL2VA-SDNQ")
        if not result.ok:
            print(result.errors)
    """

    def __init__(
        self,
        scheme_manager: SchemeManager,
        models_root: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化校验器。

        Args:
            scheme_manager: 方案管理器实例
            models_root: 模型根目录（如 models/）
            logger: 可选的日志器
        """
        self.scheme_manager = scheme_manager
        self.models_root = models_root
        self._logger = logger or logging.getLogger(__name__)

    def validate(
        self,
        scheme_id: str,
        variant_name: str,
        vram_gb: Optional[float] = None,
    ) -> ValidationResult:
        """校验方案是否适用于指定 variant。

        校验项：
            1. 方案是否存在
            2. 方案与 variant 是否匹配（通过 SchemeManager.validate_scheme_for_variant）
            3. 显存是否满足方案要求
            4. 模型文件是否存在（按 variant 的 model_subdir 检查标志文件）

        Args:
            scheme_id: 方案 ID（A / B / C / D）
            variant_name: variant 名称
            vram_gb: GPU 显存（GB），None 时自动检测

        Returns:
            ValidationResult
        """
        result = ValidationResult(ok=True, scheme_id=scheme_id, variant_name=variant_name)

        # 1. 方案是否存在
        scheme = self.scheme_manager.load_scheme(scheme_id)
        if scheme is None:
            result.ok = False
            result.errors.append(f"方案 {scheme_id} 不存在")
            return result

        # 2. 方案与 variant 是否匹配
        if not self.scheme_manager.validate_scheme_for_variant(scheme, variant_name):
            result.ok = False
            result.errors.append(
                f"方案 {scheme_id}（precision={scheme.precision}）不适用于 variant {variant_name}。"
                f"方案 {scheme_id} 适用：{scheme.applicable_variants}"
            )

        # 3. 显存检查
        if vram_gb is None:
            from .scheme_manager import _detect_gpu_vram_gb
            vram_gb = _detect_gpu_vram_gb()

        # 解析方案显存需求（如 "24GB" → 24.0，"48GB+" → 48.0）
        required_vram = self._parse_vram_required(scheme.vram_required)
        if required_vram is not None and vram_gb < required_vram:
            result.warnings.append(
                f"显存不足：方案 {scheme_id} 需要 {scheme.vram_required}，实际检测到 {vram_gb:.1f}GB。"
                f"可能影响性能或导致 OOM。"
            )

        # 4. 模型文件存在性检查（只检查标志文件，不校验大小）
        # v0.6+ 修正：scheme.model_subdir 现在是相对 model_dir 的子目录
        # （sdnq_format / diffusers_format），需要先拼上 video/minimax-h3
        # SDNQ 仓库用 modular_model_index.json，BF16 仓库用 model_index.json
        from .config import MODEL_SUBDIR
        model_dir = self.models_root / MODEL_SUBDIR / scheme.model_subdir
        if not model_dir.exists():
            result.ok = False
            result.errors.append(
                f"模型目录不存在：{model_dir}\n"
                f"请先下载方案 {scheme_id} 对应的模型（仓库：{scheme.model_repo}）"
            )
        else:
            # 按方案精度选择正确的标志文件
            # SDNQ 4bit 仓库只有 modular_model_index.json（无 model_index.json）
            # BF16 原版仓库同时有 model_index.json 和 modular_model_index.json
            if scheme.precision in ("sdnq_uint4", "sdnq_uint4_fp16"):
                marker_files = ["modular_model_index.json"]
            else:
                # BF16 优先 model_index.json，fallback modular_model_index.json
                marker_files = ["model_index.json", "modular_model_index.json"]
            marker_exists = False
            for mf in marker_files:
                marker = model_dir / mf
                if marker.exists():
                    marker_exists = True
                    break
            if not marker_exists:
                result.warnings.append(
                    f"标志文件不存在：{marker_files}（在 {model_dir}）\n"
                    f"可能模型未完整下载，建议到模型管理界面重新下载"
                )

        return result

    @staticmethod
    def _parse_vram_required(vram_str: str) -> Optional[float]:
        """解析显存需求字符串为数值。

        Args:
            vram_str: 显存需求字符串（如 "24GB" / "48GB+" / "8GB"）

        Returns:
            显存数值（GB），解析失败返回 None
        """
        if not vram_str:
            return None
        # 提取数字部分
        import re
        match = re.search(r"(\d+(?:\.\d+)?)", vram_str)
        if match:
            return float(match.group(1))
        return None
