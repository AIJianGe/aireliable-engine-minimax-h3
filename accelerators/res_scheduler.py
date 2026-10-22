"""RES multistep 二阶采样器（v1.0）。

参考 MiniMax-H3-Turbo 仓库（https://github.com/Yi-111-a/MiniMax-H3-Turbo，
Apache-2.0）的 RES（Rectified-flow Exponential Second-order multistep）实现重写，
移植到本引擎 accelerators 体系。

原理：
    官方 ``MiniMaxH3Scheduler`` 是一阶 Euler（eta=0）求解器，质量档 30 个 sigma
    点需要 29 次 DiT 前向。RES 把更新公式升级为二阶指数多步：

        x_{i+1} = (σ_next/σ_cur)·x_i + hb1·D(x_i) + hb2·D(x_{i-1})

    其中 D(x) 为 data-ward 速度还原的干净估计（x0 = x_t + σ·v），系数在
    ``set_timesteps()`` 时按 sigma 序列预计算：

        h  = ln(σ_cur/σ_next)             （本步区间的对数长度）
        c2 = ln(σ_cur/σ_prev) / h         （上一步区间与本步区间之比）
        φ1 = (e^{-h} - 1) / (-h)          （指数积分 φ 函数）
        φ2 = (φ1 - 1) / (-h)
        b1 = φ1 - φ2/c2,   b2 = φ2/c2

    首步（无 D(x_{i-1})）与末步（σ_next=0）回退 Euler，保证边界稳定；
    当 hb2 → 0 且 hb1 = 1 - σ_next/σ_cur 时，公式精确退化为 Euler。

效果（H3-Turbo 实测口径）：
    - Euler-50：49 次 DiT 调用
    - RES-21：20 次 DiT 调用（-59%，去噪阶段最高 ~2.45x 提速）
    - RES-31：30 次 DiT 调用（保守质量档）

用户决策（v1.1，2026-09-02）：
    ``install_solver`` 首版默认 ``include_audio=False`` —— 只替换视频调度器
    （``pipe.scheduler``，shift=12），音频调度器（``pipe.audio_scheduler``，
    shift=3）保持官方 Euler；视频侧实测确认质量无损后再评估双换。

设计原则（遵循 project-rules.md）：
    - 纯数学（``res_multistep_coeffs``）不依赖 torch/diffusers，便于单元验证
    - 子类 ``step()`` 先跑官方 Euler 保持状态机 / 校验语义，再替换内部步
    - ``install_solver`` 与 ``install_euler`` 对称，可来回切换
    - torch / diffusers 均延迟导入：本模块被 accelerators 包顶层引用时不引入重依赖
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Optional

__all__ = [
    "install_solver",
    "install_euler",
    "make_res_scheduler_class",
    "res_multistep_coeffs",
    "res_update",
]

# 延迟解析的基类 / 子类缓存（避免模块导入期依赖 diffusers）
_BASE_SCHEDULER_CLS: Optional[type] = None
_RES_SCHEDULER_CLS: Optional[type] = None


# ---------------------------------------------------------------------------
# 1. 纯数学：RES 系数预计算与二阶更新（无 diffusers 依赖）
# ---------------------------------------------------------------------------

def res_multistep_coeffs(
    sigmas: Sequence[float],
) -> list[Optional[tuple[float, float]]]:
    """按 sigma 序列预计算每步的 ``(h·b1, h·b2)`` 系数。

    Args:
        sigmas: 严格递减、终于 0.0 的 sigma 序列（至少 2 个点）。

    Returns:
        长度为 ``len(sigmas) - 1`` 的系数列表，与步索引一一对应：
        - 首步 / 末步 / 数值不可用步为 ``None``（回退 Euler）
        - 其余为 ``(h*b1, h*b2)`` 二阶系数元组

    Raises:
        ValueError: sigma 序列不满足严格递减 / 终于 0 / 非负有限的要求。
    """
    values = [float(value) for value in sigmas]
    if len(values) < 2:
        raise ValueError("sigma 序列至少需要 2 个点。")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("sigma 值必须为非负有限数。")
    if any(right >= left for left, right in zip(values, values[1:])):
        raise ValueError("sigma 序列必须严格递减。")
    if values[-1] != 0.0:
        raise ValueError("sigma 序列必须终于 0.0。")

    coefficients: list[Optional[tuple[float, float]]] = []
    for index, (sigma_current, sigma_next) in enumerate(zip(values, values[1:])):
        # 首步无前一帧 denoised；末步 σ_next=0 —— 均回退 Euler
        if index == 0 or sigma_current == 0.0 or sigma_next == 0.0:
            coefficients.append(None)
            continue
        sigma_previous = values[index - 1]
        h = math.log(sigma_current / sigma_next)
        c2 = math.log(sigma_current / sigma_previous) / h
        if h <= 0.0 or c2 == 0.0 or not math.isfinite(c2):
            coefficients.append(None)
            continue
        phi1 = math.expm1(-h) / (-h)
        phi2 = (phi1 - 1.0) / (-h)
        b1 = phi1 - phi2 / c2
        b2 = phi2 / c2
        coefficients.append((h * b1, h * b2))
    return coefficients


def res_update(
    sample: Any,
    denoised: Any,
    previous_denoised: Any,
    *,
    sigma_current: float,
    sigma_next: float,
    hb1: float,
    hb2: float,
) -> Any:
    """应用一次 RES 二阶更新（float32 计算，输出转回原 dtype）。

    更新公式::

        x_next = (σ_next/σ_cur)·x_t + hb1·D(x_t) + hb2·D(x_{t-1})

    半精度（fp16 / bf16）样本在 float32 上计算后转回，与官方 Euler step 的
    数值策略一致。

    Args:
        sample: 当前样本 x_t（torch.Tensor）。
        denoised: 本步干净估计 D(x_t)（torch.Tensor）。
        previous_denoised: 上一步干净估计 D(x_{t-1})（torch.Tensor）。
        sigma_current: 当前步 sigma（必须 > 0）。
        sigma_next: 下一步 sigma（必须 > 0）。
        hb1: 一阶项系数（h·b1）。
        hb2: 二阶项系数（h·b2）。

    Returns:
        下一步样本 x_{t+1}（与 sample 同 dtype / device）。

    Raises:
        ValueError: 形状不一致或 sigma 非正。
    """
    import torch  # 延迟导入：纯系数计算场景（单元验证）无需 torch

    if sample.shape != denoised.shape or sample.shape != previous_denoised.shape:
        raise ValueError("sample / denoised / previous_denoised 形状必须一致。")
    if sigma_current <= 0.0 or sigma_next <= 0.0:
        raise ValueError("RES 更新要求 sigma_current 与 sigma_next 均为正数。")

    compute_dtype = (
        torch.float32
        if sample.dtype in (torch.float16, torch.bfloat16)
        else sample.dtype
    )
    ratio = sample.new_tensor(sigma_next / sigma_current, dtype=compute_dtype)
    output = (
        ratio * sample.to(dtype=compute_dtype)
        + hb1 * denoised.to(dtype=compute_dtype)
        + hb2 * previous_denoised.to(dtype=compute_dtype)
    )
    return output.to(dtype=sample.dtype)


# ---------------------------------------------------------------------------
# 2. 调度器子类：MiniMaxH3RESMultistepScheduler
# ---------------------------------------------------------------------------

def _resolve_base_scheduler_class() -> type:
    """延迟解析官方 ``MiniMaxH3Scheduler`` 基类（带模块级缓存）。"""
    global _BASE_SCHEDULER_CLS
    if _BASE_SCHEDULER_CLS is not None:
        return _BASE_SCHEDULER_CLS
    try:
        from diffusers import MiniMaxH3Scheduler as base_cls
    except ImportError:
        try:
            from diffusers.schedulers.scheduling_minimax_h3 import (
                MiniMaxH3Scheduler as base_cls,
            )
        except ImportError as exc:
            raise RuntimeError(
                "当前 diffusers 版本不含 MiniMaxH3Scheduler，无法使用 RES 采样器。"
                "请确认引擎环境 envs/video-minimax-h3 的 diffusers 版本"
                "（本地验证基线：0.40.0.dev0）。"
            ) from exc
    _BASE_SCHEDULER_CLS = base_cls
    return base_cls


def make_res_scheduler_class() -> type:
    """构造（并缓存）RES 调度器子类 ``MiniMaxH3RESMultistepScheduler``。

    延迟构造的原因：基类 ``MiniMaxH3Scheduler`` 仅在较新的 diffusers 中存在，
    模块导入期不解析，保证本模块在无 diffusers 环境下仍可导入（纯数学可测）。
    """
    global _RES_SCHEDULER_CLS
    if _RES_SCHEDULER_CLS is not None:
        return _RES_SCHEDULER_CLS

    base = _resolve_base_scheduler_class()

    class MiniMaxH3RESMultistepScheduler(base):
        """MiniMax-H3 RES 二阶多步调度器。

        继承官方 ``MiniMaxH3Scheduler``，仅覆盖 ``set_timesteps()`` /
        ``step()`` 两个方法：

        - ``set_timesteps()``：调用官方实现生成 sigma 网格后，预计算每步
          ``(h·b1, h·b2)`` 系数（首步 / 末步为 None，回退 Euler）。
        - ``step()``：先完整跑一遍官方 Euler step（保持 ``_step_index``
          推进、入参校验、输出结构等全部官方语义），再把内部步的返回值
          替换为 RES 二阶更新结果。

        对外接口（``sigmas`` / ``timesteps`` / ``step_index`` / ``shift`` 等）
        与官方类完全一致，pipeline 无感知。
        """

        order = 2
        _is_res_multistep = True  # 标记，供 install_euler 识别需恢复的实例

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._previous_denoised: Any = None
            self._res_coefficients: list[Optional[tuple[float, float]]] = []
            self._res_sigma_values: list[float] = []

        def set_timesteps(self, *args: Any, **kwargs: Any) -> None:
            super().set_timesteps(*args, **kwargs)
            # 每次重建调度（含 set_begin_index 前的重置）都清空二阶状态
            self._previous_denoised = None
            sigma_values = [float(v) for v in self.sigmas.detach().cpu().tolist()]
            self._res_sigma_values = sigma_values
            self._res_coefficients = res_multistep_coeffs(sigma_values)

        def step(
            self,
            model_output: Any,
            timestep: Any,
            sample: Any,
            return_dict: bool = True,
        ) -> Any:
            import torch  # 延迟导入

            # 与官方 step() 相同的索引解析（不改 self._step_index，交给 super()）
            index = self._step_index
            if index is None:
                index = (
                    self.index_for_timestep(timestep)
                    if self._begin_index is None
                    else self._begin_index
                )

            # data-ward 速度还原干净估计：x0 = x_t + (1 - t)·v
            # （与官方 step 内部实现逐行对齐；t 来自 transformer 条件时间步）
            if not isinstance(timestep, torch.Tensor):
                timestep_tensor = torch.tensor(timestep, dtype=sample.dtype)
            else:
                timestep_tensor = timestep
            sigma_from_timestep = 1 - timestep_tensor.to(
                device=sample.device, dtype=sample.dtype
            )
            while sigma_from_timestep.ndim < sample.ndim:
                sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
            denoised = sample + sigma_from_timestep * model_output

            # 先跑官方 Euler：_step_index 推进 / 入参校验 / 输出结构全部保留
            base_result = super().step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                return_dict=return_dict,
            )

            # 内部步替换为 RES 二阶更新（首步 / 末步系数为 None，保留 Euler 结果）
            coefficient = (
                self._res_coefficients[index]
                if index < len(self._res_coefficients)
                else None
            )
            previous = self._previous_denoised
            if coefficient is not None and previous is not None:
                sigma_current = self._res_sigma_values[index]
                sigma_next = self._res_sigma_values[index + 1]
                hb1, hb2 = coefficient
                previous_sample = res_update(
                    sample,
                    denoised,
                    previous,
                    sigma_current=sigma_current,
                    sigma_next=sigma_next,
                    hb1=hb1,
                    hb2=hb2,
                )
                if return_dict:
                    base_result = type(base_result)(prev_sample=previous_sample)
                else:
                    base_result = (previous_sample,)

            # 记录本步干净估计，供下一步二阶项使用
            self._previous_denoised = denoised.detach()
            return base_result

    _RES_SCHEDULER_CLS = MiniMaxH3RESMultistepScheduler
    return MiniMaxH3RESMultistepScheduler


# ---------------------------------------------------------------------------
# 3. 安装 / 恢复入口：install_solver / install_euler
# ---------------------------------------------------------------------------

def _from_config(scheduler_cls: type, base_scheduler: Any) -> Any:
    """从现有调度器实例构造同类新实例（继承 config / shift）。"""
    config = getattr(base_scheduler, "config", None)
    if config is not None and hasattr(scheduler_cls, "from_config"):
        return scheduler_cls.from_config(config)
    # 兜底：直接按 shift 构造（视频 12.0 / 音频 3.0）
    return scheduler_cls(shift=float(getattr(base_scheduler, "shift", 12.0)))


def _new_res_scheduler(base_scheduler: Any) -> Any:
    """把官方 Euler 调度器替换为 RES 调度器（幂等：已是 RES 则原样返回）。"""
    res_cls = make_res_scheduler_class()
    if isinstance(base_scheduler, res_cls):
        return base_scheduler
    return _from_config(res_cls, base_scheduler)


def _new_euler_scheduler(base_scheduler: Any) -> Any:
    """把 RES 调度器恢复为官方 Euler（幂等：已是官方类则原样返回）。"""
    base_cls = _resolve_base_scheduler_class()
    if type(base_scheduler) is base_cls:
        return base_scheduler
    return _from_config(base_cls, base_scheduler)


def _replace_schedulers(pipe: Any, factory: Any, names: list[str]) -> None:
    """替换 pipe 上指定名称的调度器组件（update_components 优先、setattr 兜底）。"""
    replacements = {}
    for name in names:
        scheduler = getattr(pipe, name, None)
        if scheduler is None:
            raise AttributeError(f"H3 pipeline 没有 {name!r} 组件。")
        replacements[name] = factory(scheduler)
    update_components = getattr(pipe, "update_components", None)
    if callable(update_components):
        update_components(**replacements)
    else:
        for name, scheduler in replacements.items():
            setattr(pipe, name, scheduler)


def install_solver(
    pipe: Any,
    solver: str,
    include_audio: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """按求解器名称替换 pipeline 调度器（v1.1 决策：默认只换视频）。

    Args:
        pipe: diffusers ModularPipeline 实例（已加载组件）。
        solver: ``"res_multistep"``（RES 二阶）或 ``"euler"``（恢复官方 Euler）。
        include_audio: 是否同步替换音频调度器（``audio_scheduler``，shift=3）。
            **首版默认 False**（用户决策 v1.1）：仅替换视频调度器
            （``scheduler``，shift=12），音频保持官方 Euler，待视频侧实测
            确认质量无损后再评估双换。
        logger: 可选日志器（关键事件必记：替换了哪些组件 / shift 值）。

    Returns:
        传入的 pipe（链式调用友好）。

    Raises:
        ValueError: solver 取值未知。
        AttributeError: pipe 缺少目标调度器组件。
        RuntimeError: 本地 diffusers 不含 MiniMaxH3Scheduler（仅 RES 时）。
    """
    log = logger or logging.getLogger(__name__)

    if solver == "res_multistep":
        names = ["scheduler"] + (["audio_scheduler"] if include_audio else [])
        _replace_schedulers(pipe, _new_res_scheduler, names)
        # 关键事件必记（project-rules.md 6.5）：记录替换组件与 shift
        replaced = [
            f"{name}(shift={float(getattr(getattr(pipe, name), 'shift', 0.0)):.1f})"
            for name in names
        ]
        if not include_audio:
            replaced.append(
                "audio_scheduler 保持官方 Euler（include_audio=False，v1.1 决策）"
            )
        log.info("RES 求解器注入完成: solver=res_multistep, 组件=%s", replaced)
        return pipe

    if solver == "euler":
        return install_euler(pipe, logger=log)

    raise ValueError(f"未知的求解器: {solver!r}（可选: euler | res_multistep）")


def install_euler(
    pipe: Any,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """把所有 RES 调度器恢复为官方 Euler（已是官方类的组件不动）。

    与 ``install_solver(pipe, "euler")`` 等价；单独提供是为了与注入入口对称，
    便于 wrapper 在方案切换 / 卸载时恢复原状。

    Args:
        pipe: diffusers ModularPipeline 实例。
        logger: 可选日志器。

    Returns:
        传入的 pipe。
    """
    log = logger or logging.getLogger(__name__)
    names = [
        name
        for name in ("scheduler", "audio_scheduler")
        if getattr(pipe, name, None) is not None
    ]
    # _new_euler_scheduler 幂等：官方 Euler 组件原样返回，仅 RES 被重建
    _replace_schedulers(pipe, _new_euler_scheduler, names)
    log.info("求解器已恢复官方 Euler: 检查组件=%s", names)
    return pipe
