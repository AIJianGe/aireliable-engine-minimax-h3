"""MiniMax-H3 视频生成引擎配置。

从环境变量 + 命令行参数加载，提供引擎运行所需的全部配置。

v0.2.1 双源真相消除策略：
    - MODEL_SUBDIR 默认读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
      （由调度层 EngineManager 启动子进程时同步 manifest.models.model_layout.base）
    - 优先级：环境变量 > 硬编码 fallback

CPU offload 分级（大模型显存优化，规范：development-guide.md §37.2.7）：
    - L0: 全 GPU（所有模型常驻 GPU，需 ≥80GB 显存）
    - L1: Text Encoder 卸载（TE 编码完后移到 CPU，Transformer+VAE 常驻 GPU，需 ≥48GB 显存）
    - L2: 层级别 offload（Transformer 每层按需搬入 GPU，需 ≥24GB 显存，慢 3-5×）
    - L3: 激进 offload（全部模型在 CPU，仅计算时搬入 + VAE tiling，慢 10×+）
    - auto: 根据 GPU 显存自动选择（resolve_offload_level 解析）
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 任务 71：设备探测统一走 SDK（平台与引擎同一份事实源）
from aireliable_engine import EngineConfig as BaseEngineConfig
from aireliable_engine.device import device_info as _sdk_device_info


# ===== 常量 =====

ENGINE_ID = "minimax_h3_video_a1b2c3"
ENGINE_NAME = "MiniMax-H3 Video"
CAPABILITY = "text-to-video"
ENGINE_TYPE = "video"

# 默认端口（与 config/engines/minimax_h3_video_a1b2c3.yaml 一致）
DEFAULT_PORT = 7006

# 各生成方式（workflow）的推荐最佳采样步数
# 语义（2026-09-20 用户要求：每模型×生成方式一个推荐默认，不统一）：
#   未显式传 num_inference_steps 时，先用本表推荐值；表中没有该 workflow 才回退方案默认。
#   显式传值（含 webui 滑块/手动改、gateway 直传）优先级最高，恒生效。
#   实测基准：4 步严重模糊（不可作质量依据）/ 8 步清晰基线 ≈13min / 30 步最佳画质 ≈50min。
# 范式归属：t2v/i2v/fl2v → H3-FL2VA-4bit；ref2v/ref2va/video_edit → H3-Ref2VA-4bit。
# v1.3 单一来源（RECOMMENDED_TABLE → 派生 RECOMMENDED_STEPS）：
#   这里同时承载 webui「推荐参考表」展示所需的模型归属 / 中文方式 / 理由，
#   后续据网上推荐 / 实测补充时**只改这一处**，/recommended 端点与前端自动同步。
RECOMMENDED_TABLE: list[dict] = [
    # mode(workflow) / model(范式名,展示用) / label(中文方式) / steps / reason
    {"mode": "t2v",        "model": "H3-FL2VA",  "label": "文生视频(T2V)",  "steps": 30,
     "reason": "无参考图约束，出片即最终质量，需足量步数刻画细节/运镜"},
    {"mode": "i2v",        "model": "H3-FL2VA",  "label": "图生视频(I2V)",  "steps": 8,
     "reason": "首帧兜底主体，8 步清晰基线即可"},
    {"mode": "fl2v",       "model": "H3-FL2VA",  "label": "首尾帧(FL2V)",   "steps": 8,
     "reason": "首尾帧固定，只需渲染过渡，8 步基线"},
    {"mode": "ref2v",      "model": "H3-Ref2VA", "label": "多宫格参考(Ref2V)", "steps": 8,
     "reason": "多参考图锁身份，8 步已验证清晰可用"},
    {"mode": "ref2va",     "model": "H3-Ref2VA", "label": "多模态参考(Ref2VA)", "steps": 30,
     "reason": "图文音多模态高要求（电影感/音画同步），最佳画质"},
    {"mode": "video_edit", "model": "H3-Ref2VA", "label": "视频编辑(video_edit)", "steps": 8,
     "reason": "视频重绘，兼顾速度与细节"},
]

# 由 RECOMMENDED_TABLE 派生的 mode → steps 映射（后端 generate 兜底 + webui 自动跳步数）
RECOMMENDED_STEPS: dict[str, int] = {row["mode"]: row["steps"] for row in RECOMMENDED_TABLE}

# 模型目录布局（相对 MODELS_ROOT）
# v0.2.1：优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
_DEFAULT_MODEL_SUBDIR = "video/minimax-h3/"


def _resolve_model_subdir() -> str:
    """解析模型子目录（v0.2.1：优先读环境变量）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


# 运行时常量
MODEL_SUBDIR = _resolve_model_subdir()

# v0.2.15+ 日志统一到 logs/engines/（项目顶层）
# 旧的 LOG_SUBDIR = "logs/"（相对 data_root）已废弃

# CPU offload 级别常量
OFFLOAD_LEVELS = ("L0", "L1", "L2", "L3")
# 各级别所需的最小 GPU 显存（GB），用于 auto 自动解析
# 必须与 manifest.yaml env_config.cpu_offload.levels.Lx.min_vram_gb 保持一致
# 规范：development-guide.md §37.2.7 CPU Offload 等级规范
# L0: ≥80GB（全 GPU）、L1: ≥48GB（TE 卸载）、L2: ≥24GB（层级别 offload）、L3: <24GB（激进 offload）
_OFFLOAD_VRAM_THRESHOLDS = {
    "L0": 80,   # 全 GPU，所有模型常驻 GPU
    "L1": 48,   # Text Encoder 卸载，TE 编码完后移到 CPU
    "L2": 24,   # 层级别 offload，Transformer 每层按需搬入 GPU
}

# v0.3.5+ 设备策略合法取值（平台注入 AIRELIABLE_DEVICE_POLICY 环境变量）
#   auto    ：引擎自行决定（现状行为）
#   gpu     ：强制全 GPU，显存不足直接报错
#   gpu+cpu ：GPU 优先，显存不足时组件溢出 CPU 内存
#   cpu     ：强制 CPU（调试 / 无卡环境）
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")


def _detect_gpu_vram_gb() -> float:
    """检测当前 GPU 显存（GB），无 GPU 或检测失败返回 0.0。

    任务 71：委托 SDK ``device.device_info``（有 core 时走平台权威值，
    否则本地多源探测），不再手写 torch / nvidia-smi 双份兜底。
    """
    try:
        return _sdk_device_info(mode="auto").total_vram_gb or 0.0
    except Exception:
        return 0.0


@dataclass
class EngineConfig(BaseEngineConfig):
    """MiniMax-H3 视频生成引擎运行配置。

    通过环境变量 + 命令行参数构造。
    """

    # ===== 基础（身份 / 端口默认值由本引擎覆盖，其余公共字段见基类）=====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    # ===== CPU offload =====
    # auto: 根据 GPU 显存自动选择级别（resolve_offload_level 解析）
    # L0/L1/L2/L3: 显式指定卸载级别
    cpu_offload_level: str = "auto"

    # ===== 设备策略（v0.3.5+）=====
    # 平台通过 AIRELIABLE_DEVICE_POLICY 环境变量注入（UI 设置 > yaml > 默认 auto）。
    # 映射规则见 parse_args：gpu→L0 / gpu+cpu→auto（显存分级）/ cpu→device=cpu
    # 默认 "auto" 由基类 EngineConfig 提供。

    def __post_init__(self) -> None:
        """推导子路径。"""
        # 模型相关
        self.model_dir = self.models_root / MODEL_SUBDIR

        # 日志目录（v0.2.15+ 统一到 logs/engines/）
        self.log_dir = self.project_root / "logs" / "engines"

        # 确保运行时目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def resolve_offload_level(self) -> str:
        """解析 CPU offload 级别。

        若 cpu_offload_level 为 "auto"，则根据 GPU 显存自动选择：
            - 显存 ≥ 80GB → L0（全 GPU，所有模型常驻 GPU）
            - 显存 ≥ 48GB → L1（Text Encoder 卸载）
            - 显存 ≥ 24GB → L2（层级别 offload）
            - 显存 < 24GB / 无 GPU → L3（激进 offload）

        若已显式指定 L0/L1/L2/L3，则直接返回。

        阈值必须与 manifest.yaml env_config.cpu_offload.levels.Lx.min_vram_gb 保持一致。
        规范：development-guide.md §37.2.7 CPU Offload 等级规范

        Returns:
            解析后的 offload 级别（"L0" / "L1" / "L2" / "L3"）
        """
        if self.cpu_offload_level != "auto":
            # 显式指定，校验合法性后返回
            level = self.cpu_offload_level.upper()
            if level not in OFFLOAD_LEVELS:
                raise ValueError(
                    f"非法的 cpu_offload_level: {self.cpu_offload_level}，"
                    f"应为 auto / {(' / '.join(OFFLOAD_LEVELS))}"
                )
            return level

        # auto: 根据 GPU 显存自动选择
        vram_gb = _detect_gpu_vram_gb()
        # 按阈值从高到低匹配
        for level in ("L0", "L1", "L2"):
            if vram_gb >= _OFFLOAD_VRAM_THRESHOLDS[level]:
                return level
        # 显存不足 24GB 或无 GPU，使用 L3（激进 offload）
        return "L3"

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "device": self.device,
            "device_policy": self.device_policy,
            "log_level": self.log_level,
            "cpu_offload_level": self.cpu_offload_level,
        }


def parse_args() -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。

    优先级：命令行 > 环境变量 > 默认值
    """
    parser = argparse.ArgumentParser(description="MiniMax-H3 视频生成引擎")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda:0（默认 auto）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")
    parser.add_argument(
        "--cpu-offload-level", type=str, default="auto",
        help="CPU offload 级别：auto / L0 / L1 / L2 / L3（默认 auto）",
    )

    args = parser.parse_args()

    # 环境变量 fallback
    models_root = Path(args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models"))

    # CPU offload 级别：命令行 > 环境变量 > 默认值
    cpu_offload_level = (
        args.cpu_offload_level
        or os.environ.get("AIRELIABLE_CPU_OFFLOAD_LEVEL", "auto")
    )

    # v0.3.5+ 设备策略（平台注入 AIRELIABLE_DEVICE_POLICY）→ 映射到既有 offload 机制：
    #   gpu+cpu → cpu_offload_level=auto（显存自动分级 L0-L3，即溢出 CPU）
    #   gpu     → L0（强制全 GPU）
    #   cpu     → device=cpu（resolve_offload_level 无 GPU 自动 L3）
    #   auto    → 不干预（现状行为）
    # 仅当用户未显式配置 cpu_offload_level（值为 auto）时映射
    device_policy = os.environ.get("AIRELIABLE_DEVICE_POLICY", "auto").strip()
    if device_policy not in DEVICE_POLICIES:
        # 非法值按未设置处理（引擎侧容错，平台侧已校验）
        device_policy = "auto"
    device = args.device
    if device_policy == "gpu" and cpu_offload_level == "auto":
        cpu_offload_level = "L0"
    elif device_policy == "cpu":
        device = "cpu"

    config = EngineConfig(
        port=args.port,
        host=args.host,
        device=device,
        log_level=args.log_level,
        models_root=models_root,
        cpu_offload_level=cpu_offload_level,
        device_policy=device_policy,
    )

    return config
