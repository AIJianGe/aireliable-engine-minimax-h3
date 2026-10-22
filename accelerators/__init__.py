"""MiniMax-H3 加速方案插件包（v2.0）。

按方案配置（SchemeConfig）动态注入加速方案到 diffusers ModularPipeline：
    - SageAttention（INT8 QK + FP8 PV）：1.5-2x 加速
    - LightX2V Turbo LoRA（4/8-step 蒸馏）：6-12x 加速
    - FirstBlockCache（首块必跑+残差复用）：1.5x 加速
    - BlockSwap（自定义 diffusers 版）：把部分 transformer 块卸载到 CPU

使用方式：
    from .accelerators import apply_acceleration, restore_acceleration

    ctx = apply_acceleration(pipe, scheme, logger=logger)
    try:
        # 推理
    finally:
        restore_acceleration(ctx)

设计原则（遵循 project-rules.md 第二部分 §3）：
    - 所有 monkey-patch 必须用 try/finally 包裹，在 finally 中恢复
    - 加载/卸载对称：apply_acceleration 与 restore_acceleration 必须配对调用
    - 失败不阻断：单个加速方案注入失败时记录警告但不抛异常，让引擎仍可运行（降级到无加速）
"""

from __future__ import annotations

from .patcher import (
    AccelerationContext,
    apply_acceleration,
    restore_acceleration,
)
from .res_scheduler import install_euler, install_solver

__all__ = [
    "AccelerationContext",
    "apply_acceleration",
    "restore_acceleration",
    "install_solver",
    "install_euler",
]
