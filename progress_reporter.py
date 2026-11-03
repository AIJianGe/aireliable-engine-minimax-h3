"""SSE 进度上报工具（MiniMax-H3 视频生成专用）。

把视频生成的推理过程转换为 AIReliable 进度帧。

进度阶段：
    start             0.00          开始视频生成
    model_loading     0.05          模型已加载确认
    prompt_processing 0.10          提示词处理
    generating        0.10 - 0.85   视频帧生成中（含扩散去噪）
    vae_decoding      0.85          VAE 解码
    audio_decoding    0.90          音频解码（FL2VA / Ref2VA）
    saving            0.92          保存视频文件
    completed         1.00          完成

任务 71 P0-B：通用帧构造 / 任务上报 / 心跳 / 完成帧全部由 SDK
``BaseProgressReporter`` 提供，本文件只保留阶段常量与语义化方法。
"""

from __future__ import annotations

from typing import Any

from aireliable_engine.progress import BaseProgressReporter


# ===== 进度阶段定义 =====

class ProgressStage:
    """视频生成推理进度阶段。"""

    START = "video.start"                       # 0.00
    MODEL_READY = "video.model_ready"           # 0.05
    PROMPT_PROCESSING = "video.prompt_processing"  # 0.10
    GENERATING = "video.generating"             # 0.10 - 0.85
    VAE_DECODING = "video.vae_decoding"         # 0.85
    AUDIO_DECODING = "video.audio_decoding"     # 0.90
    SAVING = "video.saving"                     # 0.92
    MUX_WARNING = "video.mux_warning"           # 0.95
    COMPLETED = "video.completed"               # 1.00
    ERROR = "video.error"


# 进度区间
STAGE_RANGES: dict[str, tuple[float, float]] = {
    ProgressStage.START: (0.00, 0.05),
    ProgressStage.MODEL_READY: (0.05, 0.10),
    ProgressStage.PROMPT_PROCESSING: (0.10, 0.15),
    ProgressStage.GENERATING: (0.15, 0.85),
    ProgressStage.VAE_DECODING: (0.85, 0.90),
    ProgressStage.AUDIO_DECODING: (0.90, 0.92),
    ProgressStage.SAVING: (0.92, 0.95),
    ProgressStage.MUX_WARNING: (0.95, 1.00),
}


class ProgressReporter(BaseProgressReporter):
    """视频生成进度上报器（阶段语义 + SDK 通用帧）。"""

    error_type = "video_error"
    heartbeat_progress = 0.50
    heartbeat_stage = "video.heartbeat"
    heartbeat_message = "生成中（心跳）"

    # ===== 阶段便捷方法 =====

    def emit_start(self, task_id: str, description: str = "") -> None:
        """开始推理。

        Args:
            task_id: 任务 ID
            description: 任务描述（如 prompt 摘要）
        """
        self.set_task_id(task_id)
        self.emit(
            0.0,
            ProgressStage.START,
            message="开始视频生成",
            task_id=task_id,
            description=description[:200],
        )
        self._fire_running(ProgressStage.START)

    def emit_model_ready(self, variant: str, device: str) -> None:
        """模型已就绪。"""
        self.emit(
            0.05,
            ProgressStage.MODEL_READY,
            message="模型已加载",
            variant=variant,
            device=device,
        )

    def emit_prompt_processing(self, prompt: str) -> None:
        """提示词处理中。"""
        self.emit(
            0.10,
            ProgressStage.PROMPT_PROCESSING,
            message="提示词处理中",
            prompt_length=len(prompt),
        )

    def emit_generating(self, progress: float = 0.15, **extra: Any) -> None:
        """视频帧生成中。

        v0.4.x：调用方可传入 message 覆盖默认值（如 "开始扩散去噪"）。
        """
        # 用 extra.pop 提取，调用方传入的 message 优先于默认值，
        # 避免与内部默认 message 重复导致 "got multiple values for keyword argument 'message'"
        message = extra.pop("message", "生成视频帧中...")
        self.emit(progress, ProgressStage.GENERATING, message=message, **extra)

    def emit_vae_decoding(self) -> None:
        """VAE 解码中。"""
        self.emit(0.85, ProgressStage.VAE_DECODING, message="VAE 解码（潜空间 → 像素帧）")

    def emit_audio_decoding(self) -> None:
        """音频解码中（FL2VA / Ref2VA 模式）。"""
        self.emit(0.90, ProgressStage.AUDIO_DECODING, message="音频解码中")

    def emit_saving(self) -> None:
        """保存中。"""
        self.emit(0.92, ProgressStage.SAVING, message="保存视频文件")

    async def emit_mux_warning(self, message: str) -> None:
        """音视频合成警告（如时长不匹配补静音/截断）。

        在 emit_saving 之后、emit_completed 之前推送，
        前端据此显示 toast 警告提示。

        Args:
            message: 警告信息（如 "音频短于视频 1.4s，已补静音尾帧"）
        """
        await self.emit_async(0.95, ProgressStage.MUX_WARNING, message=message)

    async def emit_completed(
        self,
        task_id: str,
        result: dict,
    ) -> None:
        """完成。

        Args:
            task_id: 任务 ID
            result: 结果字典，含 video_url / audio_url / duration / fps 等字段
        """
        await super().emit_completed(
            {"task_id": task_id, **result},
            {
                "total_seconds": self.elapsed,
                "video_duration": result.get("duration", 0),
                "fps": result.get("fps", 24),
            },
        )
