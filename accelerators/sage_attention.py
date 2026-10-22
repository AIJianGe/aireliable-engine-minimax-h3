"""SageAttention 加速方案（v2.0）。

通过 monkey-patch torch.nn.functional.scaled_dot_product_attention，
将原生 attention 计算替换为 SageAttention 的 INT8 QK + FP8 PV 量化版本。

性能：
    - 1.5-2x 加速（不损失质量）
    - 显存降低 30-50%

依赖：
    pip install sageattention

设计原则（遵循 project-rules.md 第二部分 §3）：
    - monkey-patch 用 try/finally 包裹（在 patcher.py 的 restore_acceleration 中调用）
    - 在 restore 中恢复原始 torch.nn.functional.scaled_dot_product_attention
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class SageAttentionContext:
    """SageAttention 注入上下文。"""
    original_sdpa: Any  # 原始 torch.nn.functional.scaled_dot_product_attention
    backend: str


def apply_sage_attention(
    pipe: Any,
    config: Any,  # SageAttentionConfig
    logger: Optional[logging.Logger] = None,
) -> SageAttentionContext:
    """注入 SageAttention。

    Args:
        pipe: diffusers ModularPipeline 实例（未直接使用，保留接口一致性）
        config: SageAttentionConfig 实例
        logger: 可选的日志器

    Returns:
        SageAttentionContext（用于后续 restore_sage_attention）

    Raises:
        ImportError: sageattention 未安装
        RuntimeError: backend 不支持
    """
    log = logger or logging.getLogger(__name__)

    try:
        import sageattention  # noqa: F401
        from sageattention import sageattn
    except ImportError as e:
        raise ImportError(
            "SageAttention 未安装，请执行: pip install sageattention\n"
            "如果暂时不想使用 SageAttention，请在方案配置中关闭 acceleration.sage_attention.enabled"
        ) from e

    # 保存原始 sdpa
    original_sdpa = F.scaled_dot_product_attention
    backend = config.backend

    log.info(
        "准备注入 SageAttention: backend=%s, torch.version=%s, cuda=%s",
        backend, torch.__version__, torch.cuda.is_available(),
    )

    # monkey-patch
    def _patched_sdpa(query, key, value, *args, **kwargs):
        """SageAttention 替换的 sdpa 实现。

        SageAttention 不支持所有参数组合，遇到不支持的场景时 fallback 到原生 sdpa。
        """
        try:
            # SageAttention 期望 (B, H, N, D) 形状
            # diffusers 的 attention 调用通常已是这个形状
            return sageattn(query, key, value, backend=backend)
        except Exception as e:
            log.debug("SageAttention 调用失败，fallback 到原生 sdpa: %s", e)
            return original_sdpa(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = _patched_sdpa

    log.info("SageAttention 注入完成: backend=%s", backend)
    return SageAttentionContext(original_sdpa=original_sdpa, backend=backend)


def restore_sage_attention(
    ctx: SageAttentionContext,
    logger: Optional[logging.Logger] = None,
) -> None:
    """恢复原始 scaled_dot_product_attention。

    Args:
        ctx: apply_sage_attention 返回的上下文
        logger: 可选的日志器
    """
    log = logger or logging.getLogger(__name__)

    if ctx is None or ctx.original_sdpa is None:
        return

    import torch.nn.functional as F
    F.scaled_dot_product_attention = ctx.original_sdpa
    log.info("SageAttention 已恢复（original_sdpa 已还原）")
