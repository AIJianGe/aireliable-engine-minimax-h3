"""加速方案统一注入器（v2.0）。

按 SchemeConfig 决定启用哪些加速方案，并提供统一的 apply/restore 接口。

设计要点：
    - 中心化：所有加速方案的 monkey-patch 都通过本模块统一注入和恢复
    - 对称性：apply_acceleration 必须与 restore_acceleration 配对调用
    - 降级容错：单个加速方案注入失败时记录警告但不抛异常
    - 顺序：先注入 LoRA（修改权重）→ 再注入 SageAttention / FirstBlockCache / BlockSwap（修改前向计算）

遵循 project-rules.md 第二部分 §3：
    所有 monkey-patch 用 try/finally 包裹，在 finally 中恢复原始函数。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..scheme_manager import SchemeConfig


@dataclass
class AccelerationContext:
    """加速方案注入上下文，记录所有需要恢复的状态。

    Attributes:
        scheme: 关联的方案配置
        sage_ctx: SageAttention 注入上下文（None 表示未启用）
        fbc_ctx: FirstBlockCache 注入上下文
        te_ctx: TE-Speed 块级缓存注入上下文（v0.7.7+ T3.1）
        lora_ctx: LoRA 注入上下文
        bs_ctx: BlockSwap 注入上下文
        applied: 是否已注入（用于幂等性检查）
    """
    scheme: SchemeConfig
    sage_ctx: Optional[Any] = None
    fbc_ctx: Optional[Any] = None
    te_ctx: Optional[Any] = None
    lora_ctx: Optional[Any] = None
    bs_ctx: Optional[Any] = None
    applied: bool = False


def apply_acceleration(
    pipe: Any,
    scheme: SchemeConfig,
    logger: Optional[logging.Logger] = None,
) -> AccelerationContext:
    """按方案配置注入加速方案。

    Args:
        pipe: diffusers ModularPipeline 实例（已加载组件）
        scheme: 方案配置
        logger: 可选的日志器

    Returns:
        AccelerationContext（用于后续 restore_acceleration）
    """
    log = logger or logging.getLogger(__name__)
    ctx = AccelerationContext(scheme=scheme)

    if pipe is None:
        log.warning("apply_acceleration: pipe 为 None，跳过加速方案注入")
        return ctx

    accel = scheme.acceleration
    log.info(
        "开始注入加速方案: scheme=%s, sage=%s, lora=%s, fbc=%s, te_speed=%s, block_swap=%s",
        scheme.scheme,
        accel.sage_attention.enabled if accel.sage_attention else False,
        accel.lora.enabled if accel.lora else False,
        accel.first_block_cache.enabled if accel.first_block_cache else False,
        accel.te_speed_cache.enabled if accel.te_speed_cache else False,
        scheme.block_swap.enabled,
    )

    # 1. LoRA（修改权重，需在其他 patch 之前）
    if accel.lora and accel.lora.enabled:
        try:
            from .lora_loader import apply_lora
            ctx.lora_ctx = apply_lora(pipe, accel.lora, logger=log)
            log.info("LightX2V Turbo LoRA 注入成功: name=%s", accel.lora.name)
        except Exception as e:
            log.warning("LightX2V Turbo LoRA 注入失败（降级到无 LoRA）: %s", e)
            ctx.lora_ctx = None

    # 2. SageAttention（修改 attention 计算）
    if accel.sage_attention and accel.sage_attention.enabled:
        try:
            from .sage_attention import apply_sage_attention
            ctx.sage_ctx = apply_sage_attention(
                pipe, accel.sage_attention, logger=log,
            )
            log.info("SageAttention 注入成功: backend=%s", accel.sage_attention.backend)
        except Exception as e:
            log.warning("SageAttention 注入失败（降级到原生 attention）: %s", e)
            ctx.sage_ctx = None

    # 3. FirstBlockCache（修改 transformer 前向）
    # v0.7.7+ T3.1：TE-Speed 启用时替代 FirstBlockCache（TE-Speed 为 sigma 差
    # 先验 + 块级跳过，命中真正省算力；两者都 patch transformer.forward，避免叠加）
    te_speed_on = bool(accel.te_speed_cache and accel.te_speed_cache.enabled)
    if accel.first_block_cache and accel.first_block_cache.enabled and not te_speed_on:
        try:
            from .first_block_cache import apply_first_block_cache
            ctx.fbc_ctx = apply_first_block_cache(
                pipe, accel.first_block_cache, logger=log,
            )
            log.info("FirstBlockCache 注入成功: threshold=%s", accel.first_block_cache.threshold)
        except Exception as e:
            log.warning("FirstBlockCache 注入失败（降级到无缓存）: %s", e)
            ctx.fbc_ctx = None

    # 3.5 TE-Speed 块级缓存（v0.7.7+ T3.1，sigma 差先验 + 尾部块整段跳过）
    if te_speed_on:
        try:
            from .te_speed_cache import apply_te_speed_cache
            ctx.te_ctx = apply_te_speed_cache(pipe, accel.te_speed_cache, logger=log)
            log.info(
                "TeSpeedCache 注入成功: control_value=%s, window=[%s, %s]",
                accel.te_speed_cache.control_value,
                1.0 - accel.te_speed_cache.percent_2,
                1.0 - accel.te_speed_cache.percent_1,
            )
        except Exception as e:
            log.warning("TeSpeedCache 注入失败（降级到无缓存）: %s", e)
            ctx.te_ctx = None

    # 4. BlockSwap（修改 transformer 模块结构）
    if scheme.block_swap.enabled:
        try:
            from .block_swap import apply_block_swap
            ctx.bs_ctx = apply_block_swap(
                pipe, scheme.block_swap, logger=log,
            )
            log.info(
                "BlockSwap 注入成功: swap_blocks=%d, strategy=%s",
                scheme.block_swap.swap_blocks, scheme.block_swap.swap_strategy,
            )
        except Exception as e:
            log.warning("BlockSwap 注入失败（降级到无块交换）: %s", e)
            ctx.bs_ctx = None

    ctx.applied = True
    log.info("加速方案注入完成: scheme=%s", scheme.scheme)
    return ctx


def restore_acceleration(
    ctx: AccelerationContext,
    logger: Optional[logging.Logger] = None,
) -> None:
    """恢复所有加速方案的 monkey-patch。

    必须与 apply_acceleration 配对调用。即使在推理中抛异常，也应在 finally 中调用本函数。

    Args:
        ctx: apply_acceleration 返回的上下文
        logger: 可选的日志器
    """
    log = logger or logging.getLogger(__name__)

    if ctx is None or not ctx.applied:
        return

    # 逆序恢复（与注入顺序相反）
    if ctx.bs_ctx is not None:
        try:
            from .block_swap import restore_block_swap
            restore_block_swap(ctx.bs_ctx, logger=log)
            log.info("BlockSwap 已恢复")
        except Exception as e:
            log.warning("BlockSwap 恢复失败: %s", e)

    if ctx.fbc_ctx is not None:
        try:
            from .first_block_cache import restore_first_block_cache
            restore_first_block_cache(ctx.fbc_ctx, logger=log)
            log.info("FirstBlockCache 已恢复")
        except Exception as e:
            log.warning("FirstBlockCache 恢复失败: %s", e)

    if ctx.te_ctx is not None:
        try:
            from .te_speed_cache import restore_te_speed_cache
            restore_te_speed_cache(ctx.te_ctx, logger=log)
            log.info("TeSpeedCache 已恢复")
        except Exception as e:
            log.warning("TeSpeedCache 恢复失败: %s", e)

    if ctx.sage_ctx is not None:
        try:
            from .sage_attention import restore_sage_attention
            restore_sage_attention(ctx.sage_ctx, logger=log)
            log.info("SageAttention 已恢复")
        except Exception as e:
            log.warning("SageAttention 恢复失败: %s", e)

    if ctx.lora_ctx is not None:
        try:
            from .lora_loader import restore_lora
            restore_lora(ctx.lora_ctx, logger=log)
            log.info("LightX2V Turbo LoRA 已卸载")
        except Exception as e:
            log.warning("LoRA 恢复失败: %s", e)

    ctx.applied = False
    log.info("加速方案恢复完成: scheme=%s", ctx.scheme.scheme)
