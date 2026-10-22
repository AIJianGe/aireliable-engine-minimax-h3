"""简化版 Context-IR（上下文感知指令重写）。

MiniMax-H3 支持多模态输入（图片 / 视频 / 音频），
Context-IR 把用户原始 prompt + 多模态引用自动重写为带标签的结构化 prompt，
让模型更准确地理解各输入的作用。

标签规范（与 MiniMax-H3 官方文档一致）：
    <Subject N>   主体定义（N 从 1 递增）
    <Picture N>   图片引用（对齐 I2V / Ref2VA 模式的图片输入）
    <Video N>     视频引用（对齐 video_edit 模式的视频输入）
    <Audio N>     音频引用（对齐音频条件输入）

输出 prompt 结构：
    [主体定义块]
    <Subject 1>
    <Picture 1> / <Video 1> / <Audio 1> ...
    </Subject 1>

    [摘要块]
    Summary: ...

    [详细描述块]
    Detailed Description: ...

简化版实现：不调用 LLM，仅按规则拼接标签 + 模板。
完整版应调用外部 LLM（如 Qwen3）做语义重写。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PromptProcessor:
    """简化版 Context-IR 提示词处理器。

    根据输入的多模态素材（图片 / 视频 / 音频）自动构造 <Subject N> /
    <Picture N> / <Video N> / <Audio N> 标签，拼接成结构化 prompt。

    完整版 Context-IR 应调用 LLM 做语义重写，此处为规则模板版，
    保留扩展接口（_build_summary / _build_detailed_description 可被
    LLM 调用替换）。
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """初始化提示词处理器。

        Args:
            logger: 可选的日志器
        """
        self._logger = logger or logging.getLogger(__name__)

    def process(
        self,
        prompt: str,
        task_type: str,
        images: Optional[list[str]] = None,
        videos: Optional[list[str]] = None,
        audios: Optional[list[str]] = None,
    ) -> str:
        """处理原始 prompt，生成增强后的结构化 prompt。

        Args:
            prompt: 用户原始提示词
            task_type: 任务类型（t2v / i2v / fl2v / ref2v / ref2va / video_edit）
            images: 图片引用列表（URL 或路径，仅作为标签占位，不读取内容）
            videos: 视频引用列表
            audios: 音频引用列表

        Returns:
            增强后的结构化 prompt 字符串

        生成流程：
            1. _build_subject_definitions: 构造 <Subject N> 块（含 Picture/Video/Audio 标签）
            2. _build_summary: 构造摘要块
            3. _build_detailed_description: 构造详细描述块
            4. 三块拼接返回
        """
        images = images or []
        videos = videos or []
        audios = audios or []

        self._logger.info(
            "Context-IR 处理: task_type=%s, prompt_len=%d, images=%d, videos=%d, audios=%d",
            task_type, len(prompt), len(images), len(videos), len(audios),
        )

        # 无多模态输入时，直接返回原始 prompt（避免无意义的结构化）
        if not images and not videos and not audios:
            self._logger.debug("无多模态输入，跳过 Context-IR 结构化")
            return prompt

        subject_block = self._build_subject_definitions(images, videos, audios)
        summary_block = self._build_summary(task_type, prompt)
        detailed_block = self._build_detailed_description(prompt)

        # 三块拼接（用空行分隔）
        parts = [p for p in (subject_block, summary_block, detailed_block) if p.strip()]
        enhanced_prompt = "\n\n".join(parts)

        self._logger.debug(
            "Context-IR 输出: enhanced_len=%d (orig_len=%d)",
            len(enhanced_prompt), len(prompt),
        )
        return enhanced_prompt

    def _build_subject_definitions(
        self,
        images: list[str],
        videos: list[str],
        audios: list[str],
    ) -> str:
        """构造 <Subject N> 主体定义块。

        每个主体包含相关的图片 / 视频 / 音频引用标签。
        简化版：所有引用归入 Subject 1（单一主体场景）。
        完整版应基于多模态理解分配主体归属。

        Args:
            images: 图片引用列表
            videos: 视频引用列表
            audios: 音频引用列表

        Returns:
            主体定义块字符串（含 <Subject 1> ... </Subject 1> 标签）
        """
        lines: list[str] = ["<Subject 1>"]

        # 图片引用
        for i, _img in enumerate(images, start=1):
            lines.append(f"<Picture {i}>")

        # 视频引用
        for i, _vid in enumerate(videos, start=1):
            lines.append(f"<Video {i}>")

        # 音频引用
        for i, _aud in enumerate(audios, start=1):
            lines.append(f"<Audio {i}>")

        lines.append("</Subject 1>")
        return "\n".join(lines)

    def _build_summary(self, task_type: str, prompt: str) -> str:
        """构造摘要块。

        根据任务类型生成简短的摘要描述，帮助模型快速定位生成目标。

        Args:
            task_type: 任务类型
            prompt: 原始提示词

        Returns:
            摘要块字符串（"Summary: ..."）
        """
        # 任务类型 → 摘要模板
        type_descriptions = {
            "t2v": "Text to Video generation",
            "i2v": "Image to Video generation",
            "fl2v": "Full Latent to Video Audio generation (with audio)",
            "ref2v": "Reference to Video Audio generation (with audio)",
            "ref2va": (
                "Multimodal Reference to Video Audio generation "
                "(image/video/audio references, with audio)"
            ),
            "video_edit": "Video editing based on reference video",
        }
        type_desc = type_descriptions.get(task_type, "Video generation")

        # 摘要取 prompt 前 100 字符（避免过长）
        prompt_summary = prompt.strip()[:100]
        if len(prompt.strip()) > 100:
            prompt_summary += "..."

        return f"Summary: {type_desc}. {prompt_summary}"

    def _build_detailed_description(self, prompt: str) -> str:
        """构造详细描述块。

        把原始 prompt 完整放入，让模型获取完整上下文。
        完整版应基于 LLM 扩写为更具描述性的段落（含画面构图 / 镜头运动 / 风格等）。

        Args:
            prompt: 原始提示词

        Returns:
            详细描述块字符串（"Detailed Description: ..."）
        """
        return f"Detailed Description: {prompt.strip()}"
