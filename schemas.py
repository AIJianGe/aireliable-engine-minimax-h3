"""MiniMax-H3 视频生成引擎 Pydantic 数据模型。

扩展 SDK 的 VariantInfo / InferRequest，添加视频专属字段。

视频生成模式（task_type）：
    - t2v:         Text to Video（文本生成视频）
    - i2v:         Image to Video（图片 + 文本生成视频）
    - fl2v:        Full Latent to Video（文本直接生成含音频的视频）
    - ref2v:       Reference to Video（参考图 + 文本生成含音频的视频）
    - ref2va:      Multimodal Reference to Video+Audio（多模态参考：图片/视频/音频
                   混合参考 + 文本生成含音频视频，Ref2VA 范式；按上传组合自动判定
                   场景——多图参考/视频参考/对口型（图+音频）/音色参考等；
                   音频参考不能单独使用，必须配对至少 1 张图片或 1 个视频；
                   上限：图 ≤9 / 视频 ≤3 / 音频 ≤3）
    - video_edit:  视频编辑（参考视频 + 文本编辑）
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from aireliable_engine import VariantInfo


# ===== 视频生成模式枚举 =====
# fl2v: Full Latent to Video（文本直接生成含音频的视频，FL2VA 范式）
# ref2v: Reference to Video（参考图 + 文本生成含音频的视频，Ref2VA 范式）
# ref2va: Multimodal Reference to Video+Audio（多模态参考：图/视频/音频混合，
#         Ref2VA 范式，音频必须配对图/视频；图≤9 / 视频≤3 / 音频≤3）
# i2v: Image to Video（图片 + 文本生成视频，无音频）
# t2v: Text to Video（纯文本生成视频，无音频）
# video_edit: 视频编辑（参考视频 + 文本编辑）
VIDEO_MODES = ("t2v", "i2v", "fl2v", "ref2v", "ref2va", "video_edit")


# ===== CPU offload 级别（与 config.py 的 OFFLOAD_LEVELS 保持一致） =====
OFFLOAD_LEVELS = ("L0", "L1", "L2", "L3")


class VideoVariantInfo(VariantInfo):
    """视频变体能力声明（扩展 SDK VariantInfo）。

    在标准字段之外补充视频特有信息：分辨率 / 帧率 / 时长 / 是否含音频。
    Pydantic model_dump() 会序列化全部字段，前端可读取这些扩展字段
    渲染视频专属 UI。
    """

    model_config = ConfigDict(extra="allow")

    resolution: str = Field(
        "768x432",
        description="默认输出分辨率（宽 x 高）",
    )
    fps: int = Field(24, ge=1, le=60, description="默认帧率")
    duration: int = Field(5, ge=1, le=60, description="默认时长（秒）")
    max_duration: int = Field(60, ge=1, description="最大支持时长（秒）")
    supports_audio: bool = Field(
        True,
        description="是否支持音频生成（FL2VA / Ref2VA 支持，T2V / I2V 不支持）",
    )


class VideoInputs(BaseModel):
    """视频推理输入参数（放入 InferRequest.inputs）。

    所有模式共用此 schema，根据 task_type 字段决定使用哪些参数。
    """

    model_config = ConfigDict(extra="allow")

    # ===== 必填 =====
    prompt: str = Field(..., description="正向提示词")

    # ===== 任务类型与负向提示词 =====
    negative_prompt: str = Field("", description="负向提示词")
    task_type: str = Field(
        "t2v",
        description=(
            "生成模式：t2v（文生视频）/ i2v（图生视频）/ fl2v（首尾帧生音频视频）/ "
            "ref2v（多宫格参考图）/ ref2va（多模态参考：图/视频/音频混合，音频必须配对"
            "图/视频，上限 图≤9/视频≤3/音频≤3）/ video_edit（视频编辑）"
        ),
    )

    # ===== 视频参数 =====
    duration: int = Field(5, ge=1, le=60, description="视频时长（秒）")
    fps: int = Field(24, ge=1, le=60, description="帧率")
    # v0.6.5+: 放宽分辨率下限到 128，支持 432x240 等低清档（44-回归测试要求）
    width: int = Field(768, ge=128, le=1920, description="视频宽度（8 的倍数）")
    height: int = Field(432, ge=128, le=1920, description="视频高度（8 的倍数）")

    # ===== 推理参数 =====
    # v0.6.4+ 修复：默认 None，让 _apply_scheme_defaults 使用方案默认值（如方案 A 的 4 步）
    # 如果显式传入，则使用传入值（覆盖方案默认值）
    num_inference_steps: Optional[int] = Field(
        None, ge=1, le=200, description="推理步数（去噪步数，None=使用方案默认值）"
    )

    # ===== 画质方案（quality_scheme，45-专项，与模型精度档位 scheme 正交） =====
    # 注意：此处 A/B/C/D 是【画质增强路线】，与 scheme_manager 的 A/B/C/D（显存/精度档位）
    # 是两套独立概念，字段名 quality_scheme 与之明确区分，勿混用。
    # 取值为独立 token，用 "+" 组合，按管线顺序生效（生成内D → A超分 → C脸修复）：
    #   "B"           高分辨率直出（768×432/864×480，画质优先）—— 默认
    #   "A"           低清生成 + 后期超分（复用 video_up_enlarge）
    #   "C"           追加脸部修复后处理
    #   "D"           采样增强（RF Restart / Tail σ / STG，生成内生效）
    #   组合示例       "A+C" / "A+C+D" / "B+D" / "B+C"
    #   "none"        裸生成，禁用全部画质增强（基线对照）
    # 约束：A 与 B 互斥（同给告警并按 A 处理，均由分辨率档驱动）
    quality_scheme: str = Field(
        "B",
        description=(
            "画质方案（45-专项，与模型精度档位 scheme 正交）："
            "B=高分辨率直出(默认) / A=后期超分 / C=脸修复 / D=采样增强；"
            "可用 '+' 组合，如 'A+C'；'none' 为基线对照；A 与 B 互斥"
        ),
    )
    guidance_scale: Optional[float] = Field(
        None, ge=0.0, le=20.0, description="CFG 引导强度（None=使用方案默认值）"
    )
    seed: int = Field(-1, ge=-1, description="随机种子（-1=随机）")

    # ===== CPU offload 级别 =====
    # auto: 根据 GPU 显存自动选择（config.resolve_offload_level 解析）
    # L0/L1/L2/L3: 显式指定卸载级别
    # 注意：此字段会覆盖引擎启动时的 cpu_offload_level 配置
    cpu_offload_level: str = Field(
        "auto",
        description=f"CPU offload 级别：auto / {' / '.join(OFFLOAD_LEVELS)}",
    )


class VideoStatusExtra(BaseModel):
    """视频引擎状态附加信息。"""

    device: str = Field("cpu", description="当前推理设备")
    vram_used: float = Field(0.0, description="已用显存（MB）")
    offload_level: str = Field("L1", description="当前 CPU offload 级别")
    loaded_variant: Optional[str] = Field(None, description="已加载的模型变体")
    loaded_loras: list[str] = Field(
        default_factory=list, description="已加载的 LoRA 列表"
    )
