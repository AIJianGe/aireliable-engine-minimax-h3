"""FirstBlockCache 加速方案（v2.1）。

移植自 ComfyUI 的 FirstBlockCache 机制：
    - 第一个 transformer 块总是执行（强制，保证至少有一帧完整推理）
    - 后续块检查残差（输出与上次差异），如果残差小于阈值则直接复用上次结果
    - 加速约 1.5x，质量损失极小

实现方式：
    - monkey-patch transformer 的 forward 方法
    - 在 forward 入口记录上次输出，下次调用时比较残差
    - 残差 = ||output_new - output_old|| / ||output_old||
    - 残差 < threshold 时返回 output_old（跳过本次计算）

v2.1 修复：MiniMax-H3 transformer.forward 返回 tuple (noise_pred, audio_noise_pred)，
而非单 Tensor。原始 v2.0 直接 `output.detach()` 导致
`AttributeError: 'tuple' object has no attribute 'detach'`。
本版本递归处理 tuple/list/单 Tensor 三种返回形式。

设计原则（遵循 project-rules.md 第二部分 §3）：
    - monkey-patch 用 try/finally 包裹（在 patcher.py 的 restore_acceleration 中调用）
    - 在 restore 中恢复原始 forward
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch


def _detach_recursive(output: Any) -> Any:
    """递归 detach 输出，支持 Tensor / tuple / list / dict / namedtuple。

    Args:
        output: transformer.forward 的返回值，可能是：
            - torch.Tensor：单张量（标准 diffusers transformer）
            - tuple：MiniMax-H3 等多任务模型返回 (noise_pred, audio_noise_pred)
            - namedtuple：diffusers 某些模型返回 BaseModelOutput 等
            - list / dict：其他可能的结构

    Returns:
        结构与输入一致，所有 Tensor 已 detach（脱离计算图）

    v2.1.1 修复：处理 namedtuple（如 diffusers TransformerOutput），
    用 _make 构造而非 type(output)(...)，避免 TypeError。
    """
    if isinstance(output, torch.Tensor):
        return output.detach()
    if isinstance(output, tuple):
        # namedtuple 检查：有 _fields 和 _make 属性
        tuple_type = type(output)
        if hasattr(tuple_type, "_make") and hasattr(tuple_type, "_fields"):
            # namedtuple：用 _make 构造（接受迭代器）
            return tuple_type._make(_detach_recursive(item) for item in output)
        # 普通 tuple
        return tuple(_detach_recursive(item) for item in output)
    if isinstance(output, list):
        return [_detach_recursive(item) for item in output]
    if isinstance(output, dict):
        return {k: _detach_recursive(v) for k, v in output.items()}
    # 其他类型（int/None/...）原样返回
    return output


def _compute_residual(
    new_output: Any,
    old_output: Any,
    device: Any,
) -> float:
    """计算 new_output 相对 old_output 的相对残差。

    对于 tuple/list，逐元素计算范数后求和；对于单 Tensor，直接计算。
    任意一个元素类型不匹配或形状不一致时，返回 +inf 强制视为"未命中"。

    Args:
        new_output: 当前 forward 输出
        old_output: 上次缓存的输出（已 detach）
        device: 目标设备（新输出的设备）

    Returns:
        相对残差标量；不可比较时返回 float('inf')
    """
    try:
        if isinstance(new_output, torch.Tensor) and isinstance(old_output, torch.Tensor):
            old = old_output.to(device)
            new = new_output
            base = old.norm().clamp(min=1e-6)
            residual = (new - old).norm()
            return (residual / base).item()

        if isinstance(new_output, (tuple, list)) and isinstance(old_output, (tuple, list)):
            if len(new_output) != len(old_output):
                return float("inf")
            total_residual = 0.0
            total_base = 0.0
            for n, o in zip(new_output, old_output):
                if isinstance(n, torch.Tensor) and isinstance(o, torch.Tensor):
                    o_dev = o.to(n.device)
                    total_residual += (n - o_dev).norm().item()
                    total_base += o_dev.norm().clamp(min=1e-6).item()
                else:
                    return float("inf")
            if total_base <= 0:
                return float("inf")
            return total_residual / total_base

        # 类型不匹配（一个 Tensor，一个 tuple 等）
        return float("inf")
    except Exception:
        return float("inf")


@dataclass
class FirstBlockCacheContext:
    """FirstBlockCache 注入上下文。"""
    transformer: Any
    original_forward: Any
    threshold: float
    # 运行时缓存（v2.1: 可能是 Tensor 或 tuple of Tensors）
    _last_output: Optional[Any] = None
    _cache_hits: int = 0
    _cache_misses: int = 0


def apply_first_block_cache(
    pipe: Any,
    config: Any,  # FirstBlockCacheConfig
    logger: Optional[logging.Logger] = None,
) -> Optional[FirstBlockCacheContext]:
    """注入 FirstBlockCache。

    Args:
        pipe: diffusers ModularPipeline 实例（取 pipe.transformer 或 pipe.transformer_ref）
        config: FirstBlockCacheConfig 实例（含 threshold）
        logger: 可选的日志器

    Returns:
        FirstBlockCacheContext，未找到 transformer 返回 None
    """
    log = logger or logging.getLogger(__name__)

    # 找到 transformer（FL2VA 用 transformer，Ref2VA 用 transformer_ref）
    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "transformer_ref", None)
    if transformer is None:
        log.warning("FirstBlockCache: 未找到 transformer/transformer_ref 组件，跳过")
        return None

    original_forward = transformer.forward
    threshold = config.threshold

    log.info(
        "准备注入 FirstBlockCache: transformer=%s, threshold=%.4f",
        type(transformer).__name__, threshold,
    )

    ctx = FirstBlockCacheContext(
        transformer=transformer,
        original_forward=original_forward,
        threshold=threshold,
    )

    def _patched_forward(*args, **kwargs):
        """FirstBlockCache 替换的 forward 实现。

        v2.1: 支持 tuple 返回值（MiniMax-H3 transformer 返回
        (noise_pred, audio_noise_pred)）。
        v2.1.3: 重构异常处理，分离 original_forward 和缓存处理：
            - original_forward 失败：直接 re-raise（不重试，避免循环抛出）
            - 缓存处理失败（detach/compute_residual）：降级返回 output（不缓存）
        避免旧版 except 块内再次调用 original_forward 导致同样的异常传播。
        """
        # 第一步：执行 original_forward（失败时直接传播，不重试）
        try:
            output = original_forward(*args, **kwargs)
        except Exception as _fwd_e:
            log.warning(
                "FirstBlockCache: original_forward 失败（直接传播，不重试）: %s: %s",
                type(_fwd_e).__name__, _fwd_e,
            )
            ctx._last_output = None
            raise

        # 调试日志：记录 output 类型（首次调用时）
        if ctx._cache_misses + ctx._cache_hits == 0:
            if isinstance(output, tuple):
                _elem_types = [type(item).__name__ for item in output]
                log.info(
                    "FirstBlockCache 首次调用: output=tuple(len=%d), elem_types=%s",
                    len(output), _elem_types,
                )
            else:
                log.info(
                    "FirstBlockCache 首次调用: output=%s",
                    type(output).__name__,
                )

        # 第二步：处理缓存（失败时降级返回 output，不影响推理）
        try:
            # 第一次调用或上次输出不可用时，强制执行并缓存
            if ctx._last_output is None:
                ctx._last_output = _detach_recursive(output)
                ctx._cache_misses += 1
                return output

            # 后续调用：计算残差（支持 tuple/Tensor）
            rel_residual = _compute_residual(output, ctx._last_output, None)

            if rel_residual < threshold:
                # 残差小于阈值：返回上次输出（跳过本次的下游计算）
                ctx._cache_hits += 1
                # 更新缓存（用新输出，避免连续复用导致偏差累积）
                ctx._last_output = _detach_recursive(output)
                log.debug(
                    "FirstBlockCache 命中: residual=%.4f < threshold=%.4f, hits=%d, misses=%d",
                    rel_residual, threshold, ctx._cache_hits, ctx._cache_misses,
                )
                # 返回上次输出（避免下游消费本次结果）
                return ctx._last_output
            else:
                # 残差大于阈值：使用新输出，更新缓存
                ctx._last_output = _detach_recursive(output)
                ctx._cache_misses += 1
                return output
        except Exception as _fbc_e:
            # v2.1.3 降级保护：缓存处理异常不应中断推理
            # output 已在第一步成功计算，直接返回（不缓存）
            log.warning(
                "FirstBlockCache 缓存处理异常，降级到无缓存模式（output type=%s）: %s: %s",
                type(output).__name__, type(_fbc_e).__name__, _fbc_e,
            )
            ctx._last_output = None  # 清空缓存，下次重新开始
            return output

    transformer.forward = _patched_forward

    log.info("FirstBlockCache 注入完成")
    return ctx


def restore_first_block_cache(
    ctx: FirstBlockCacheContext,
    logger: Optional[logging.Logger] = None,
) -> None:
    """恢复原始 forward。

    Args:
        ctx: apply_first_block_cache 返回的上下文
        logger: 可选的日志器
    """
    log = logger or logging.getLogger(__name__)

    if ctx is None or ctx.transformer is None:
        return

    ctx.transformer.forward = ctx.original_forward

    total = ctx._cache_hits + ctx._cache_misses
    hit_rate = (ctx._cache_hits / total * 100) if total > 0 else 0.0
    log.info(
        "FirstBlockCache 已恢复: hits=%d, misses=%d, hit_rate=%.1f%%",
        ctx._cache_hits, ctx._cache_misses, hit_rate,
    )

    # 清理缓存
    ctx._last_output = None
