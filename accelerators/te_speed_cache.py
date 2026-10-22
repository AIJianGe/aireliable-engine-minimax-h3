"""TE-Speed 块级缓存加速（v0.7.7+ T3.1，移植原理，非拷贝代码）。

原理参考 HELPMEEADICE/TE-Speed-MiniMaxH3-OSS（LGPL-3.0，仅借鉴思路与
视频教程中的公开参数，代码独立实现）：
    - MiniMax-H3 DiT 共 50 层（transformer.transformer_blocks）
    - 尾部 start_block_ratio 比例的块（默认 50%）在相邻去噪 step 之间复用输出：
        相邻 step 的 sigma 差值 < control_value（默认 0.12）且 sigma 处于缓存
        窗口 [1 - percent_2, 1 - percent_1]（默认 [0.1, 0.9]）时，本次 forward
        **跳过尾部段全部块的逐 block 计算**，直接复用上一步缓存的段输出。
    - 命中时"整段跳过"：段内每个 block 的 patched forward 都返回同一份缓存
      段输出，等价于 norm_out 直接消费上一步的段输出（真正省算力）。
    - 显存：仅缓存 1 份段输出（hidden_states 大小）；可配置 device="cpu" 降显存
      （命中时搬回 GPU 一次）。

sigma 来源：diffusers MiniMaxH3Transformer3DModel.forward 的 timestep 参数
即当前噪声水平（[0, 1] 无缩放），与调度器 sigma 等价，无需读取 scheduler。

与现有 first_block_cache 的区别：
    - first_block_cache：输出残差**后验** + 整 transformer 输出级（命中不省算力）
    - te_speed_cache：sigma 差**先验** + 块级跳过（命中直接不计算，真正省算力）

设计原则（遵循 project-rules.md 第二部分 §3）：
    - monkey-patch 用 try/finally 包裹（patcher.py 统一 apply/restore 配对调用）
    - 任一环节异常降级到无缓存，不影响推理
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class TeSpeedCacheContext:
    """TE-Speed 缓存注入上下文。"""
    transformer: Any
    original_forward: Any
    blocks: Any
    original_block_forwards: dict = field(default_factory=dict)
    start_idx: int = 0
    config: Any = None
    # ---- 运行时状态（跨 step）----
    _prev_sigma: Optional[float] = None
    _step_count: int = 0
    _hit_this_step: bool = False
    _tail_output: Optional[torch.Tensor] = None   # 上一步段输出缓存（GPU 或 CPU）
    _gpu_cache: Optional[torch.Tensor] = None     # 本 step 命中时的目标设备副本
    _tail_updated: bool = False                   # 本 step 段尾块是否已更新缓存
    _hits: int = 0
    _misses: int = 0
    _skipped_blocks: int = 0


def _extract_sigma(args: tuple, kwargs: dict) -> Optional[float]:
    """从 transformer.forward 参数提取 sigma（timestep，[0,1] 无缩放）。

    签名（diffusers MiniMaxH3Transformer3DModel.forward）：
        forward(hidden_states, audio_hidden_states, encoder_hidden_states,
                timestep, timestep_indices, token_tags, position_ids,
                video_indices, audio_indices, text_indices, ...)
    timestep 是第 4 个位置参数（index 3），也可能走 kwargs。
    packed sequence 可能含多个 timestep（conditioning rows），取 max 代表当前步。
    """
    timestep = kwargs.get("timestep")
    if timestep is None and len(args) > 3:
        timestep = args[3]
    if timestep is None:
        return None
    try:
        return float(timestep.max())
    except Exception:
        return None


def apply_te_speed_cache(
    pipe: Any,
    config: Any,  # TeSpeedCacheConfig
    logger: Optional[logging.Logger] = None,
) -> Optional[TeSpeedCacheContext]:
    """注入 TE-Speed 块级缓存。

    Args:
        pipe: diffusers ModularPipeline 实例（取 pipe.transformer 或 transformer_ref）
        config: TeSpeedCacheConfig 实例
        logger: 可选的日志器

    Returns:
        TeSpeedCacheContext；未找到 transformer 或配置未启用返回 None
    """
    log = logger or logging.getLogger(__name__)

    if config is None or not getattr(config, "enabled", False):
        return None

    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "transformer_ref", None)
    if transformer is None:
        log.warning("TeSpeedCache: 未找到 transformer/transformer_ref 组件，跳过")
        return None

    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks or len(blocks) == 0:
        log.warning("TeSpeedCache: transformer 无 transformer_blocks 属性，跳过")
        return None

    n_blocks = len(blocks)
    start_idx = max(0, min(n_blocks - 1, int(n_blocks * float(config.start_block_ratio))))
    log.info(
        "准备注入 TeSpeedCache: transformer=%s, blocks=%d, start_idx=%d（尾部 %d 块参与缓存）, "
        "control_value=%.3f, window=[%.2f, %.2f], device=%s",
        type(transformer).__name__, n_blocks, start_idx, n_blocks - start_idx,
        float(config.control_value),
        1.0 - float(config.percent_2), 1.0 - float(config.percent_1),
        config.device,
    )

    ctx = TeSpeedCacheContext(
        transformer=transformer,
        original_forward=transformer.forward,
        blocks=blocks,
        start_idx=start_idx,
        config=config,
    )

    # ---- 1. 外层 forward：维护 sigma 状态与命中判定 ----
    original_forward = transformer.forward

    def _patched_forward(*args, **kwargs):
        sigma = _extract_sigma(args, kwargs)
        ctx._step_count += 1
        ctx._tail_updated = False

        hit = False
        if (
            sigma is not None
            and ctx._prev_sigma is not None
            and ctx._tail_output is not None
            and start_idx < n_blocks
        ):
            sigma_diff = abs(sigma - ctx._prev_sigma)
            in_window = (1.0 - float(config.percent_2)) <= sigma <= (1.0 - float(config.percent_1))
            hit = sigma_diff < float(config.control_value) and in_window

        ctx._prev_sigma = sigma
        ctx._hit_this_step = hit
        if hit:
            ctx._hits += 1
            # 命中：确保缓存段输出在目标设备（device="cpu" 时本 step 只搬一次）
            try:
                _dev = next(transformer.parameters()).device
            except Exception:
                _dev = None
            if _dev is not None and str(_dev).startswith("cuda"):
                ctx._gpu_cache = ctx._tail_output.to(_dev)
            else:
                ctx._gpu_cache = ctx._tail_output
            log.debug(
                "TeSpeedCache 命中: sigma=%.4f, prev=%.4f, diff=%.4f < %.3f, step=%d",
                sigma, ctx._prev_sigma, abs(sigma - ctx._prev_sigma),
                float(config.control_value), ctx._step_count,
            )
        else:
            ctx._misses += 1

        try:
            output = original_forward(*args, **kwargs)
        finally:
            ctx._hit_this_step = False
            ctx._gpu_cache = None
            # miss 但段尾块未执行（如 gradient_checkpointing 路径）：不缓存本次值，
            # 清空旧缓存防止跨 sigma 误用
            if not hit and not ctx._tail_updated:
                ctx._tail_output = None
        return output

    transformer.forward = _patched_forward

    # ---- 2. 块级 forward：命中时整段跳过 ----
    def _make_patched(idx: int, orig: Any):
        def _inner(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
            if ctx._hit_this_step:
                # 命中：整段跳过，返回缓存段输出（段内后续块同样返回它）
                ctx._skipped_blocks += 1
                return ctx._gpu_cache if ctx._gpu_cache is not None else ctx._tail_output
            out = orig(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
            if idx == n_blocks - 1:
                # 段尾块：记录段输出（供下次 step 复用）
                ctx._tail_output = out.detach()
                ctx._tail_updated = True
            return out
        return _inner

    for idx in range(start_idx, n_blocks):
        block = blocks[idx]
        ctx.original_block_forwards[idx] = block.forward
        block.forward = _make_patched(idx, block.forward)

    log.info("TeSpeedCache 注入完成（尾部 %d 块）", n_blocks - start_idx)
    return ctx


def restore_te_speed_cache(
    ctx: Optional[TeSpeedCacheContext],
    logger: Optional[logging.Logger] = None,
) -> None:
    """恢复原始 forward（与 apply_te_speed_cache 配对调用）。"""
    log = logger or logging.getLogger(__name__)

    if ctx is None or ctx.transformer is None:
        return

    for idx, orig in ctx.original_block_forwards.items():
        try:
            ctx.blocks[idx].forward = orig
        except Exception as e:
            log.warning("TeSpeedCache 恢复 block %d 失败: %s", idx, e)

    ctx.transformer.forward = ctx.original_forward

    total = ctx._hits + ctx._misses
    hit_rate = (ctx._hits / total * 100) if total > 0 else 0.0
    log.info(
        "TeSpeedCache 已恢复: hits=%d, misses=%d, hit_rate=%.1f%%, skipped_blocks=%d",
        ctx._hits, ctx._misses, hit_rate, ctx._skipped_blocks,
    )

    ctx._tail_output = None
    ctx._gpu_cache = None
