"""BlockSwap 显存优化（v2.0）。

自定义实现的 diffusers 版块交换，把部分 transformer 块卸载到 CPU。
diffusers 无原生 BlockSwap 支持，本模块通过 monkey-patch transformer 的 forward
实现块级别的 GPU ↔ CPU 搬移。

工作原理：
    - transformer 通常由 N 个 transformer 块组成（MiniMax-H3 约 40 个）
    - BlockSwap 把其中 swap_blocks 个块常驻 CPU
    - forward 时按需把 CPU 块搬到 GPU 计算，计算完再搬回 CPU
    - swap_strategy='even' 时均匀分布（避免连续块都在 CPU）

性能：
    - 显存占用降低 swap_blocks / total_blocks 比例
    - 速度损失 2-3x（PCIe 搬移开销）

设计原则（遵循 project-rules.md 第二部分 §3）：
    - monkey-patch 用 try/finally 包裹（在 patcher.py 的 restore_acceleration 中调用）
    - 在 restore 中恢复原始 forward
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class BlockSwapContext:
    """BlockSwap 注入上下文。"""
    transformer: Any
    original_forward: Any
    swap_blocks: int
    swap_strategy: str
    # 被卸载到 CPU 的块索引列表
    cpu_block_indices: list[int] = field(default_factory=list)
    # 原始块设备（用于恢复）
    original_devices: dict[int, str] = field(default_factory=dict)


def apply_block_swap(
    pipe: Any,
    config: Any,  # BlockSwapConfig
    logger: Optional[logging.Logger] = None,
) -> Optional[BlockSwapContext]:
    """注入 BlockSwap。

    Args:
        pipe: diffusers ModularPipeline 实例
        config: BlockSwapConfig 实例（含 swap_blocks / swap_strategy）
        logger: 可选的日志器

    Returns:
        BlockSwapContext，未找到 transformer 返回 None
    """
    log = logger or logging.getLogger(__name__)

    # 找到 transformer
    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "transformer_ref", None)
    if transformer is None:
        log.warning("BlockSwap: 未找到 transformer 组件，跳过")
        return None

    # 找到 transformer 块列表
    # MiniMax-H3 transformer 可能的结构：
    #   - transformer.transformer_blocks（标准 diffusers 命名）
    #   - transformer.blocks（部分实现）
    #   - transformer.h（GPT 风格命名）
    blocks = (
        getattr(transformer, "transformer_blocks", None)
        or getattr(transformer, "blocks", None)
        or getattr(transformer, "h", None)
    )
    if blocks is None:
        log.warning("BlockSwap: 未找到 transformer 块列表，跳过")
        return None

    total_blocks = len(blocks)
    swap_blocks = min(config.swap_blocks, total_blocks - 1)  # 至少保留 1 块在 GPU
    if swap_blocks <= 0:
        log.warning("BlockSwap: swap_blocks=%d 无效，跳过", config.swap_blocks)
        return None

    # 计算被卸载到 CPU 的块索引
    if config.swap_strategy == "even":
        # 均匀分布：每隔 total/swap_blocks 个块取一个
        step = max(1, total_blocks // swap_blocks)
        cpu_indices = list(range(0, total_blocks, step))[:swap_blocks]
    else:  # "tail"
        # 尾部连续：把最后 swap_blocks 个块卸载
        cpu_indices = list(range(total_blocks - swap_blocks, total_blocks))

    log.info(
        "准备注入 BlockSwap: transformer=%s, total_blocks=%d, swap=%d, strategy=%s, cpu_indices=%s",
        type(transformer).__name__, total_blocks, swap_blocks, config.swap_strategy, cpu_indices,
    )

    # 保存原始设备和 forward
    original_forward = transformer.forward
    original_devices = {}

    # 把 CPU 块移到 CPU
    for idx in cpu_indices:
        block = blocks[idx]
        try:
            original_devices[idx] = str(next(block.parameters()).device)
            block.to("cpu")
        except Exception as e:
            log.warning("BlockSwap: 块 %d 移动到 CPU 失败: %s", idx, e)

    ctx = BlockSwapContext(
        transformer=transformer,
        original_forward=original_forward,
        swap_blocks=swap_blocks,
        swap_strategy=config.swap_strategy,
        cpu_block_indices=cpu_indices,
        original_devices=original_devices,
    )

    # monkey-patch forward，在调用前后搬移块
    def _patched_forward(*args, **kwargs):
        """BlockSwap 替换的 forward 实现。

        策略：在每次 forward 调用前，把当前需要计算的块搬到 GPU，
        计算完再搬回 CPU。diffusers 的 forward 内部会按顺序调用每个块。
        为了实现这一点，我们 hook 每个块的 forward 方法。
        """
        # 临时 hook 每个被卸载块的 forward，实现 on-demand 搬移
        hooks = []
        for idx in cpu_indices:
            block = blocks[idx]
            original_block_forward = block.forward

            def _make_patched_forward(orig_fwd, blk):
                def _patched_block_forward(*b_args, **b_kwargs):
                    # 搬到 GPU
                    try:
                        blk.to("cuda", non_blocking=True)
                    except Exception as e:
                        log.debug("BlockSwap: 块搬到 GPU 失败（继续 CPU 计算）: %s", e)
                    # 调用原始 forward
                    output = orig_fwd(*b_args, **b_kwargs)
                    # 搬回 CPU
                    try:
                        blk.to("cpu", non_blocking=True)
                    except Exception as e:
                        log.debug("BlockSwap: 块搬回 CPU 失败: %s", e)
                    return output
                return _patched_block_forward

            patched = _make_patched_forward(original_block_forward, block)
            block.forward = patched
            hooks.append((block, original_block_forward))

        try:
            # 调用 transformer 的原始 forward
            output = original_forward(*args, **kwargs)
        finally:
            # 恢复每个块的 forward
            for block, orig_fwd in hooks:
                block.forward = orig_fwd

        return output

    transformer.forward = _patched_forward

    log.info(
        "BlockSwap 注入完成: cpu_blocks=%d/%d (%.0f%%)",
        len(cpu_indices), total_blocks, len(cpu_indices) / total_blocks * 100,
    )
    return ctx


def restore_block_swap(
    ctx: BlockSwapContext,
    logger: Optional[logging.Logger] = None,
) -> None:
    """恢复原始 forward 并把所有块搬回 GPU（如果可能）。

    Args:
        ctx: apply_block_swap 返回的上下文
        logger: 可选的日志器
    """
    log = logger or logging.getLogger(__name__)

    if ctx is None or ctx.transformer is None:
        return

    # 恢复原始 forward
    ctx.transformer.forward = ctx.original_forward

    # 把 CPU 块搬回原始设备
    blocks = (
        getattr(ctx.transformer, "transformer_blocks", None)
        or getattr(ctx.transformer, "blocks", None)
        or getattr(ctx.transformer, "h", None)
    )
    if blocks is not None:
        for idx in ctx.cpu_block_indices:
            if idx < len(blocks):
                block = blocks[idx]
                orig_device = ctx.original_devices.get(idx, "cuda:0")
                try:
                    block.to(orig_device)
                except Exception as e:
                    log.debug("BlockSwap: 块 %d 恢复到 %s 失败: %s", idx, orig_device, e)

    log.info(
        "BlockSwap 已恢复: 恢复了 %d 个块到原始设备",
        len(ctx.cpu_block_indices),
    )
