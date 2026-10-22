"""官方 Comfy-Org Turbo LoRA 加载器（v3.0，适配自定义 ModularPipeline 模式）。

v3.0 重写（2026-08-26，T3.2）：从占位级 PEFT 实现改为**手写旁路 adapter**，
适配 ComfyUI 格式官方 turbo LoRA（4/8-step 蒸馏）到本地 diffusers
ModularPipeline 的 SDNQ 量化层。

为什么不用 PEFT / 数学合并（分析见 ANALYSIS-TASKS/
MINIMAX_H3_RTX8000_FP16_TURBO_LORA_ANALYSIS.md + 任务表 T1.2 key diff）：
    - 本地 SDNQ 权重为 uint4 打包 + 分组 scale/zero_point（一维扁平数组），
      ΔW 数学合并进原始权重不可行
    - PEFT 对 SDNQLinear 自定义量化层的 target_modules 匹配不可靠
    - 旁路 lora_a/lora_b 子模块注册为量化层的**子模块**，`.to(device)`
      时自动跟随 L2/L3 offload 的组件移动，无 device mismatch 风险

ComfyUI key → 本地模块映射规则（key diff 实测定稿）：
    diffusion_model.blocks.{i}.attn.qkv_proj    → transformer_blocks.{i}.attn.to_q/to_k/to_v
                                                  （fused 拆分：lora_B 按 out 维均分三段，
                                                   假定顺序 q/k/v，lora_A 三份共用）
    diffusion_model.blocks.{i}.attn.out_proj    → transformer_blocks.{i}.attn.to_out.0
    diffusion_model.blocks.{i}.mlp.fc1          → transformer_blocks.{i}.ff.net.0.proj
    diffusion_model.blocks.{i}.mlp.fc2          → transformer_blocks.{i}.ff.net.2
    diffusion_model.token_refiner.blocks.{i}.*  → token_refiner.refiner_blocks.{i}.*

LoRA 文件结构（官方 turbo 实测）：每模块 3 key（alpha 标量 / lora_A / lora_B），
rank：qkv_proj=384，其余=128；有效 scale = alpha / r。

遵循 project-rules.md：失败降级不阻断引擎启动；注入/卸载对称。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# 旁路 adapter 的子模块属性名（挂到量化层实例上，随 .to(device) 移动）
_LORA_A_ATTR = "_aireliable_lora_a"
_LORA_B_ATTR = "_aireliable_lora_b"

# 旁路计算精度：fp16（T 系方案激活精度；fp16 GEMM 在 sm_70+ 有 Tensor Core 路径；
# forward 包装内自动做 dtype 桥接，bf16 激活下也能正确运行）
_BYPASS_DTYPE = torch.float16


@dataclass
class LoRAContext:
    """LoRA 旁路注入上下文。"""

    transformer: Any
    lora_name: str
    lora_path: str
    adapter_name: Optional[str] = None
    scale: float = 1.0
    # 已注入旁路的模块列表——restore 时逐个恢复
    wrapped: list = field(default_factory=list)
    injected: int = 0   # 成功注入的模块数（>0 才算注入成功）
    skipped: int = 0    # 跳过数（模块缺失/维度不匹配）


def _mod_name(module: nn.Module) -> str:
    """尽力获取模块名（用于日志）。"""
    return getattr(module, "_aireliable_lora_mod_path", type(module).__name__)


# ---------------------------------------------------------------------------
# key 解析与映射
# ---------------------------------------------------------------------------

def _parse_lora_keys(tensors: dict) -> dict[str, dict]:
    """把 safetensors 的 key 按 ComfyUI 模块路径分组。

    Returns:
        {comfyui_module_path: {"alpha": float, "A": Tensor, "B": Tensor}}
        其中 comfyui_module_path 如 "blocks.0.attn.qkv_proj"（已剥掉
        diffusion_model. 前缀）。
    """
    groups: dict[str, dict] = {}
    for key, tensor in tensors.items():
        m = re.match(r"^diffusion_model\.(.+)\.(lora_A\.weight|lora_B\.weight|alpha)$", key)
        if not m:
            continue
        mod_path, leaf = m.group(1), m.group(2)
        entry = groups.setdefault(mod_path, {})
        if leaf == "alpha":
            entry["alpha"] = float(tensor.item()) if tensor.numel() == 1 else 1.0
        elif leaf == "lora_A.weight":
            entry["A"] = tensor
        else:
            entry["B"] = tensor
    return groups


def _map_targets(comfyui_mod: str) -> list[tuple[str, Optional[int]]]:
    """ComfyUI 模块路径 → [(本地模块路径, 段索引)]。

    fused qkv_proj 拆成 to_q/to_k/to_v 三份（B 按 out 维均分三段，假定
    q/k/v 顺序；A 三份共用）。段索引 0/1/2，None 表示整块映射。
    token_refiner 与主 blocks 同规则（blocks → refiner_blocks）。
    """
    # token_refiner：blocks → refiner_blocks（与主 blocks 同样的 attn/mlp 拆分规则）
    m = re.match(r"^token_refiner\.blocks\.(\d+)\.(.+)$", comfyui_mod)
    if m:
        tb = f"token_refiner.refiner_blocks.{m.group(1)}"
        return _map_rest(tb, m.group(2))

    m = re.match(r"^blocks\.(\d+)\.(.+)$", comfyui_mod)
    if not m:
        return []
    tb = f"transformer_blocks.{m.group(1)}"
    return _map_rest(tb, m.group(2))


def _map_rest(tb: str, rest: str) -> list[tuple[str, Optional[int]]]:
    """按模块后缀（attn/mlp）映射目标（主 blocks 与 token_refiner 共用）。"""
    if rest == "attn.qkv_proj":
        # fused qkv → 三个独立模块，B 均分三段（q/k/v 顺序）
        return [
            (f"{tb}.attn.to_q", 0),
            (f"{tb}.attn.to_k", 1),
            (f"{tb}.attn.to_v", 2),
        ]
    if rest == "attn.out_proj":
        return [(f"{tb}.attn.to_out.0", None)]
    if rest == "mlp.fc1":
        return [(f"{tb}.ff.net.0.proj", None)]
    if rest == "mlp.fc2":
        return [(f"{tb}.ff.net.2", None)]
    return []


# ---------------------------------------------------------------------------
# 注入与恢复
# ---------------------------------------------------------------------------

def _inject_bypass(
    module: nn.Module,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    alpha: float,
    user_scale: float,
    log: logging.Logger,
) -> bool:
    """向单个（量化）Linear 层注入 LoRA 旁路。

    旁路公式：y = orig(x) + (alpha / r) * user_scale * lora_b(lora_a(x))
    """
    if hasattr(module, _LORA_A_ATTR):
        log.warning("模块已有 LoRA 旁路（重复注入防护），跳过: %s", _mod_name(module))
        return False

    r, in_features = a_weight.shape
    out_features = b_weight.shape[0]

    # 维度一致性校验（模块有 in/out 元信息时）
    m_in = getattr(module, "in_features", None)
    m_out = getattr(module, "out_features", None)
    if m_in is not None and m_in != in_features:
        log.warning("LoRA 维度不匹配（in）: 模块=%s, 期望=%s, LoRA=%s",
                    _mod_name(module), m_in, in_features)
        return False
    if m_out is not None and m_out != out_features:
        log.warning("LoRA 维度不匹配（out）: 模块=%s, 期望=%s, LoRA=%s",
                    _mod_name(module), m_out, out_features)
        return False

    # v0.7.1 修复（设备）：nn.Linear 默认在 CPU 创建，add_module 不会迁移设备——
    # L2 下 transformer 主干已直接加载到 GPU，而 lora 权重留在 CPU，
    # 旁路 matmul 报 "mat2 is on cpu" 后静默回退纯主干（turbo LoRA 完全失效）。
    # 注入时显式对齐宿主层设备。
    try:
        host_device = next(module.parameters()).device
    except StopIteration:
        host_device = a_weight.device

    # v0.7.2 修复（dtype）：nn.Linear 参数默认 fp32，weight.copy_(fp16源) 中
    # copy_ 会把源转回目标 dtype → lora 权重实际是 fp32，
    # 与 fp16 输入 matmul 报 "Half != float"（此前被设备错误掩盖，旁路 100% 回退）。
    # 先把 Linear 整体转到 fp16 + 宿主设备，再 copy_（目标 fp16，源 bf16 安全降精度）。
    lora_a = nn.Linear(in_features, r, bias=False).to(device=host_device, dtype=_BYPASS_DTYPE)
    lora_b = nn.Linear(r, out_features, bias=False).to(device=host_device, dtype=_BYPASS_DTYPE)
    with torch.no_grad():
        lora_a.weight.copy_(a_weight)
        lora_b.weight.copy_(b_weight)

    module.add_module(_LORA_A_ATTR, lora_a)
    module.add_module(_LORA_B_ATTR, lora_b)

    combined_scale = (alpha / r) * user_scale

    orig_forward = module.forward

    def _lora_forward(x, *args, **kwargs):
        y = orig_forward(x, *args, **kwargs)
        try:
            # 惰性设备守卫：offload 策略动态移动设备时兜底（L3 逐组件移动等场景）
            if lora_a.weight.device != x.device:
                lora_a.to(x.device)
                lora_b.to(x.device)
            delta = lora_b(lora_a(x.to(_BYPASS_DTYPE)))
            y = y + (combined_scale * delta).to(y.dtype)
        except RuntimeError as e:
            # 旁路计算失败（如非预期输入形态）不应打断主干推理
            log.warning("LoRA 旁路计算失败（回退纯主干）: %s: %s", _mod_name(module), e)
        return y

    module.forward = _lora_forward
    module._aireliable_lora_orig_forward = orig_forward  # restore 用
    return True


def apply_lora(
    pipe: Any,
    config: Any,  # LoRAConfig
    logger: Optional[logging.Logger] = None,
) -> Optional[LoRAContext]:
    """注入官方 Turbo LoRA 到 transformer（旁路 adapter 方式）。

    范式感知：pipe 上哪个 transformer 属性非 None 就用哪个分区，
    Ref2VA 分区优先用 config.ref2v_path（官方 ref2v turbo 是独立文件）。
    """
    log = logger or logging.getLogger(__name__)

    # 1. 定位 transformer 分区（引擎只加载当前范式的分区，另一个为 None）
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        path_str = config.path
        paradigm = "FL2VA"
    else:
        transformer = getattr(pipe, "transformer_ref", None)
        path_str = (getattr(config, "ref2v_path", "") or config.path)
        paradigm = "Ref2VA"
    if transformer is None:
        log.warning("LoRA 加载: 未找到 transformer/transformer_ref 组件，跳过")
        return None
    if not path_str:
        log.warning("LoRA 加载: path 为空，跳过")
        return None

    # 2. 解析 LoRA 文件路径（相对路径多候选）
    lora_path = Path(path_str)
    if not lora_path.is_absolute():
        engine_dir = Path(__file__).parent.parent
        candidates = [
            engine_dir / path_str,                                   # 引擎目录
            engine_dir / "loras" / lora_path.name,
            Path("models") / "video" / "minimax-h3" / path_str,      # 平台模型目录
            Path("models") / "video" / "minimax-h3" / "loras" / lora_path.name,
        ]
        for cand in candidates:
            if cand.exists():
                lora_path = cand
                break
        else:
            log.warning(
                "LoRA 加载: 文件不存在 path=%s, 候选=%s, 跳过（请到模型管理界面下载 turbo LoRA）",
                path_str, [str(c) for c in candidates],
            )
            return None

    log.info(
        "准备注入 Turbo LoRA: name=%s, path=%s, paradigm=%s, scale=%.2f",
        config.name, lora_path, paradigm, config.scale,
    )

    # 3. 读取 LoRA 权重并按 ComfyUI 模块分组
    try:
        from safetensors.torch import load_file
        tensors = load_file(str(lora_path))
    except Exception as e:
        log.warning("LoRA 文件读取失败: %s: %s", lora_path, e)
        return None

    groups = _parse_lora_keys(tensors)
    if not groups:
        log.warning("LoRA 文件无有效 key（非 ComfyUI diffusion_model 格式）: %s", lora_path)
        return None

    # 4. 建立本地模块查找表
    named = dict(transformer.named_modules())

    ctx = LoRAContext(
        transformer=transformer,
        lora_name=config.name,
        lora_path=str(lora_path),
        adapter_name=config.name,
        scale=config.scale,
    )

    # 5. 逐组注入
    for comfyui_mod, entry in sorted(groups.items()):
        a = entry.get("A")
        b = entry.get("B")
        if a is None or b is None:
            ctx.skipped += 1
            continue
        alpha = entry.get("alpha", 1.0)

        targets = _map_targets(comfyui_mod)
        if not targets:
            ctx.skipped += 1
            continue

        for local_path, seg_idx in targets:
            module = named.get(local_path)
            if module is None:
                ctx.skipped += 1
                log.warning("LoRA 目标模块不存在，跳过: %s", local_path)
                continue
            module._aireliable_lora_mod_path = local_path  # 日志用

            # fused qkv：B 均分三段取对应段；其余整块
            if seg_idx is not None:
                out_dim = b.shape[0] // 3
                seg_b = b[seg_idx * out_dim: (seg_idx + 1) * out_dim, :]
            else:
                seg_b = b

            if _inject_bypass(module, a, seg_b, alpha, config.scale, log):
                ctx.injected += 1
                ctx.wrapped.append(module)

    # 6. 结果判定（禁止静默零注入——注入 0 个模块视为失败）
    if ctx.injected == 0:
        log.error(
            "Turbo LoRA 注入 0 个模块（全部跳过，skipped=%d）——文件与模型结构不匹配: %s",
            ctx.skipped, lora_path,
        )
        return None

    log.info(
        "Turbo LoRA 注入成功: name=%s, paradigm=%s, 注入模块数=%d, 跳过=%d, 路径=%s",
        config.name, paradigm, ctx.injected, ctx.skipped, lora_path,
    )
    return ctx


def restore_lora(
    ctx: Optional[LoRAContext],
    logger: Optional[logging.Logger] = None,
) -> None:
    """卸载 LoRA（恢复原 forward、移除旁路子模块）。与 apply_lora 对称。"""
    log = logger or logging.getLogger(__name__)

    if ctx is None:
        return

    restored = 0
    for module in ctx.wrapped:
        try:
            orig = getattr(module, "_aireliable_lora_orig_forward", None)
            if orig is not None:
                module.forward = orig
                del module._aireliable_lora_orig_forward
            if hasattr(module, _LORA_A_ATTR):
                del module._aireliable_lora_a
            if hasattr(module, _LORA_B_ATTR):
                del module._aireliable_lora_b
            if hasattr(module, "_aireliable_lora_mod_path"):
                del module._aireliable_lora_mod_path
            restored += 1
        except Exception as e:
            log.warning("LoRA 旁路恢复失败（模块随 pipe 释放，可忽略）: %s", e)

    log.info(
        "Turbo LoRA 已卸载: name=%s, 恢复模块数=%d/%d",
        ctx.lora_name, restored, len(ctx.wrapped),
    )
