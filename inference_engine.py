"""MiniMax-H3 推理引擎（v0.6+ 多方案版）。

从 minimax_h3_wrapper.py 分离出来的推理逻辑，负责：
    - 按方案配置解析推理参数（步数、分辨率、CFG 等）
    - 调用 wrapper.generate() 执行实际推理
    - 多模态输入处理（图片/视频/音频路径收集）
    - 推理结果元数据增强

设计原则（遵循 project-rules.md）：
    - 推理逻辑与加载逻辑分离：wrapper 只负责 load/unload，InferenceEngine 负责推理
    - 方案参数优先级：request 参数 > scheme 默认 > wrapper 默认
    - 错误处理：捕获推理异常，包装为 RuntimeError 抛给上层

用法：
    engine = InferenceEngine(wrapper, logger=logger)
    result = engine.generate(
        mode="fl2v",
        prompt="一只猫在弹钢琴",
        scheme=scheme_config,
        duration=5,
        seed=42,
        image="/path/to/first_frame.png",
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .minimax_h3_wrapper import MiniMaxH3Wrapper
from .config import RECOMMENDED_STEPS
from .scheme_manager import SchemeConfig


class InferenceEngine:
    """MiniMax-H3 推理引擎（多方案版）。

    职责：
        1. 从方案配置解析推理参数（步数/分辨率/CFG/调度器等）
        2. 收集多模态输入（image/last_image/references/video）
        3. 调用 wrapper.generate() 执行实际扩散推理
        4. 返回增强的推理结果（含方案元数据）

    不负责：
        - 模型加载/卸载（由 wrapper 负责）
        - HTTP 端点处理（由 engine.py 负责）
        - 视频保存（由 video_utils.save_video 负责）
    """

    def __init__(
        self,
        wrapper: MiniMaxH3Wrapper,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化推理引擎。

        Args:
            wrapper: MiniMax-H3 模型封装器（已实现 load/unload/generate）
            logger: 可选的日志器
        """
        self.wrapper = wrapper
        self._logger = logger or logging.getLogger(__name__)

    def generate(
        self,
        mode: str,
        prompt: str,
        scheme: Optional[SchemeConfig] = None,
        **kwargs: Any,
    ) -> dict:
        """执行视频生成推理（v0.6+ 多方案版）。

        参数优先级（从高到低）：
            1. 调用方显式传入的 kwargs
            2. 方案配置（scheme.inference）的默认值
            3. wrapper.generate() 的内置默认值

        Args:
            mode: 生成模式：
                - "t2v":         Text to Video（纯文本生成视频，无音频）
                - "i2v":         Image to Video（图片 + 文本生成视频，无音频）
                - "fl2v":        Full Latent to Video（首帧 + 文本生成含音频视频）
                - "ref2v":       Reference to Video（参考图 + 文本生成含音频视频）
                - "video_edit":  视频编辑（参考视频 + 文本生成视频）
            prompt: 提示词（已通过 PromptProcessor 处理）
            scheme: v0.6+ 方案配置（用于解析推理参数默认值）
            **kwargs: 推理参数（覆盖方案默认值）：
                - negative_prompt: 负向提示词（MiniMax-H3 不支持，忽略）
                - duration: 时长（秒），5-15
                - fps: 帧率（MiniMax-H3 固定 24fps，其他值忽略）
                - width / height: 分辨率（必须为 32 的倍数）
                - num_inference_steps: 推理步数
                - guidance_scale: CFG 引导强度
                - seed: 随机种子（-1=随机）
                - image: 参考图片路径/URL（I2V / Ref2VA 模式）
                - last_image: 尾帧图片路径/URL（FL2V 模式）
                - references: 参考图片路径列表（Ref2V 模式）
                - video: 参考视频路径/URL（video_edit 模式）

        Returns:
            dict 含：
                - video_frames: 视频帧序列 (T, H, W, C) uint8
                - audio_data: 音频数据 (channels, samples) float32（无音频时为 None）
                - metadata: 元数据字典（含方案 ID、模式、seed、步数等）

        Raises:
            RuntimeError: 模型未加载 / 推理失败
        """
        if not self.wrapper.is_loaded:
            raise RuntimeError("模型未加载，请先调用 wrapper.load()")

        # v1.3：未显式传步数时，先注入「工作流推荐最佳步数」（优先级高于方案默认、低于显式传值）。
        # 语义：webui 选择工作流即跳到推荐值；gateway/API 没传 num_inference_steps 时也走这里，
        #       只有本表无该 workflow 时才回退到方案默认步数。
        if not kwargs.get("num_inference_steps"):
            rec = RECOMMENDED_STEPS.get(mode)
            if rec:
                kwargs["num_inference_steps"] = rec
                self._logger.info("未传 num_inference_steps，使用工作流 %s 推荐步数 %d", mode, rec)

        # v0.6+：从方案配置解析推理参数默认值（kwargs 优先级更高，不覆盖）
        if scheme is not None:
            self._apply_scheme_defaults(scheme, kwargs)
            self._logger.info(
                "推理参数（方案 %s 默认值已合并）: steps=%s, cfg=%s, resolution=%s",
                scheme.scheme,
                kwargs.get("num_inference_steps"),
                kwargs.get("guidance_scale"),
                f"{kwargs.get('width', '?')}x{kwargs.get('height', '?')}",
            )

        # 45-专项：应用画质方案（在方案默认值之后，最终确定分辨率档）
        qs_tokens = self._apply_quality_scheme(kwargs)

        start_time = time.time()
        try:
            result = self.wrapper.generate(mode=mode, prompt=prompt, **kwargs)
        except Exception as e:
            self._logger.exception("MiniMax-H3 推理失败: mode=%s", mode)
            raise RuntimeError(f"推理失败: {e}") from e

        elapsed = time.time() - start_time

        # v0.6+：增强元数据（加入方案 ID）
        metadata = result.get("metadata", {})
        metadata["scheme_id"] = scheme.scheme if scheme else self.wrapper.loaded_scheme_id
        metadata["quality_scheme"] = "+".join(sorted(qs_tokens)) or "none"
        # 45-专项：回写"实际生效"参数（quality_scheme 抬升/解析之后的值），
        # 供引擎 completed 帧 → 任务列表展示，便于用户判断（请求值 ≠ 生效值）。
        metadata["effective_width"] = kwargs.get("width")
        metadata["effective_height"] = kwargs.get("height")
        metadata["num_inference_steps"] = kwargs.get("num_inference_steps")
        metadata["guidance_scale"] = kwargs.get("guidance_scale")
        metadata["inference_engine_elapsed"] = elapsed
        result["metadata"] = metadata

        return result

    # ---- 45-专项：画质方案（quality_scheme）分辨率档常量 ----
    # B（高分辨率直出）与 A（高分辨率直出 + 后期超分）共用同一主渲染基线：
    #   768×448（768=24×32、448=14×32，均为 32 倍数）。
    # ⚠️ 实测教训（45-测试B）：原 768×432 的 432 非 32 倍数，被 diffusers
    #    MiniMaxH3 before_denoise 拒绝（`height and width must be multiples of 32`）。
    # ⚠️ 实测教训（45-测试A v1.1）：早期 A 用 608×352 低清主渲染 + RealESRGAN 通用超分，
    #    8 步 turbo 在低分辨率下人物五官根本没被生成出来，通用超分只能放大已有像素、
    #    无法凭空补出清晰五官 → 短剧人物不可辨。故 v1.2 起 A 与 B 主渲染同档（768×448，
    #    超分 2x → 1536×896），超分成为真增益而非补洞。
    _QS_HIGH_BASE = (768, 448)

    @staticmethod
    def _snap32(v: int) -> int:
        """向上取整到最近的 32 倍数（H3 分辨率硬性约束）。"""
        return int((int(v) + 31) // 32 * 32)

    def _apply_quality_scheme(self, kwargs: dict) -> set:
        """应用画质方案（45-专项），在 _apply_scheme_defaults 之后调用。

        核心约束在【分辨率档】：
            - B / A（均高分辨率直出 768×448，默认含 B）：把 W/H 抬升到 768×448 基线，
              请求已达标则保持——从信息量根源解决"448/608 低清必糊"。
              差异：A 额外在生成后做后期超分（2x → 1536×896），B 不做。
            - C / D / none：本方法不改分辨率（C 是后处理、D 是采样侧，后续阶段处理），
              仅在日志/元数据中记录意图。

        Args:
            kwargs: 推理参数字典（原地修改，同时 pop 掉 quality_scheme）

        Returns:
            解析后的画质方案 token 集合（供后续阶段与元数据使用）
        """
        raw = str(kwargs.pop("quality_scheme", "B") or "B").strip().lower()
        if raw == "none":
            self._logger.info("画质方案=none（基线对照，禁用全部画质增强）")
            return set()

        known = {"a", "b", "c", "d"}
        tokens = {t.strip() for t in raw.split("+") if t.strip()}
        tokens = {t for t in tokens if t in known} or {"b"}

        # A ⊥ B 互斥约束：v1.2 起 A 与 B 共用同一主渲染分辨率档（差异仅在是否超分），
        # 同给时按 A 处理（保留超分阶段）
        if "a" in tokens and "b" in tokens:
            self._logger.warning(
                "画质方案同时含 A 与 B（主渲染同档，差异仅在是否超分），按 A（含超分）处理"
            )
            tokens.discard("b")

        # B 与 A 共用同一主渲染分辨率抬升（A 额外依赖后处理超分，见 quality_postprocess）
        if "b" in tokens or "a" in tokens:
            scheme_label = "A" if "a" in tokens else "B"
            w_in = int(kwargs.get("width") or 0)
            h_in = int(kwargs.get("height") or 0)
            base_w, base_h = self._QS_HIGH_BASE
            if w_in < base_w or h_in < base_h:
                w, h = max(w_in, base_w), max(h_in, base_h)
                w, h = self._snap32(w), self._snap32(h)
                self._logger.warning(
                    "画质方案=%s 高分辨率直出：分辨率 %sx%s 抬升至 %sx%s（低于 %sx%s 基线会糊）",
                    scheme_label, w_in, h_in, w, h, base_w, base_h,
                )
                kwargs["width"], kwargs["height"] = w, h
            else:
                self._logger.info(
                    "画质方案=%s 高分辨率直出：分辨率 %sx%s（已达标，保持）", scheme_label, w_in, h_in,
                )

        self._logger.info("画质方案=%s", "+".join(sorted(tokens)))
        return tokens

    def _apply_scheme_defaults(
        self,
        scheme: SchemeConfig,
        kwargs: dict,
    ) -> None:
        """将方案配置的推理参数作为默认值注入 kwargs（不覆盖已有值）。

        Args:
            scheme: 方案配置
            kwargs: 推理参数字典（原地修改）
        """
        inf = scheme.inference

        # 步数（方案 A 用 LightX2V 4-step LoRA 时为 4，方案 D 为 30）
        # v0.6.4+ 修复：inputs.num_inference_steps 默认 None 时会被传入 kwargs，
        # 需额外检查 None，否则不会回退到方案默认值（导致 steps=50 而非 4）。
        # v0.8+ turbo 锁定：turbo 方案（T4/T8，LoRA 名含 "turbo"）为蒸馏步数，
        # 必须强制使用方案步数。请求传入的 steps 仅作提示，超出蒸馏范围直接忽略并告警，
        # 否则 50 步会覆盖 8 步，采样器超出蒸馏范围导致 NaN 与音频退化（实测根因）。
        is_turbo = bool(
            scheme.acceleration.lora
            and scheme.acceleration.lora.enabled
            and "turbo" in (scheme.acceleration.lora.name or "").lower()
        )
        if is_turbo:
            requested = kwargs.get("num_inference_steps")
            if requested and int(requested) != inf.num_inference_steps:
                self._logger.warning(
                    "turbo 方案 %s 锁定推理步数为 %d（请求值 %s 被忽略，超出蒸馏范围会导致 NaN）",
                    scheme.scheme, inf.num_inference_steps, requested,
                )
            kwargs["num_inference_steps"] = inf.num_inference_steps
        else:
            if not kwargs.get("num_inference_steps"):
                kwargs["num_inference_steps"] = inf.num_inference_steps

        # CFG 引导强度
        if not kwargs.get("guidance_scale"):
            kwargs["guidance_scale"] = inf.guidance_scale

        # 默认分辨率（如 "768x432" → width=768, height=432）
        if not kwargs.get("width") and not kwargs.get("height") and inf.default_resolution:
            w, h = self._parse_resolution(inf.default_resolution)
            if w and h:
                kwargs["width"] = w
                kwargs["height"] = h

        # 默认时长（方案配置的 default_duration）
        if not kwargs.get("duration"):
            kwargs["duration"] = inf.default_duration

        # FPS（方案配置的 default_fps，但 MiniMax-H3 固定 24fps，wrapper 会忽略）
        if not kwargs.get("fps"):
            kwargs["fps"] = inf.default_fps

        # v1.1+ sigma shift（对齐官方 turbo 蒸馏参数，diffusers 每请求可覆盖）
        # 官方：8 步 turbo shift=12/3（视频/音频），4 步 768p 版 shift=6/3
        # 未显式传 shift 时注入方案默认值，wrapper.generate() 据此 set_shift。
        if not kwargs.get("flow_shift"):
            kwargs["flow_shift"] = inf.flow_shift
        if not kwargs.get("audio_flow_shift"):
            kwargs["audio_flow_shift"] = inf.audio_flow_shift

        # v0.6.6+ 修复：MiniMax-H3 VAE 要求 width/height 必须是 32 的倍数。
        # 方案默认分辨率 "768x432" 中 432 不是 32 的倍数（432/32=13.5），
        # 会导致 diffusers 抛 "height and width must be multiples of 32" 错误。
        # 这里对所有 width/height（无论来源）对齐到 32 的倍数。
        #
        # v0.6.7+ 修复：由【向下取整】改为【就近取整 + 保底下限】，杜绝静默缩水（44-回归测试实测）。
        #   根因：向下取整会把 432x240 悄悄压成 416x224（latent 仅 13x7=91 token），
        #         画面出现严重格子/块状伪影（t2v）或全图失焦糊脸（i2v），且非步数所致。
        #   处理：就近取整（四舍五入到 32 倍数）保证不缩水 —— 432x240 → 448x256（14x8=112 token，可出画面）。
        #         已合法的 32 倍数分辨率（如 448/608/768...）保持不变。
        #   保底：MIN_W=448 / MIN_H=256 —— 更小的 latent 实测黑屏/糊/块状，不足信。
        MIN_W, MIN_H = 448, 256
        w_in = kwargs.get("width")
        h_in = kwargs.get("height")
        if w_in:
            w_aligned = max(((int(w_in) + 16) // 32) * 32, MIN_W)
            if w_aligned != w_in:
                self._logger.warning(
                    "width 就近对齐 32 倍数并保底: %d → %d（原值 %d 不是 32 倍数或低于下限）",
                    w_in, w_aligned, w_in,
                )
            kwargs["width"] = w_aligned
        if h_in:
            h_aligned = max(((int(h_in) + 16) // 32) * 32, MIN_H)
            if h_aligned != h_in:
                self._logger.warning(
                    "height 就近对齐 32 倍数并保底: %d → %d（原值 %d 不是 32 倍数或低于下限）",
                    h_in, h_aligned, h_in,
                )
            kwargs["height"] = h_aligned

    @staticmethod
    def _parse_resolution(resolution: str) -> tuple[int, int]:
        """解析分辨率字符串为 (width, height)。

        Args:
            resolution: 分辨率字符串（如 "768x432" / "1280x720"）

        Returns:
            (width, height) 元组，解析失败返回 (0, 0)
        """
        if not resolution:
            return (0, 0)
        try:
            parts = resolution.lower().split("x")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        except (ValueError, AttributeError):
            pass
        return (0, 0)

    def collect_multimodal_inputs(
        self,
        mode: str,
        images: Optional[list] = None,
        videos: Optional[list] = None,
        audios: Optional[list] = None,
    ) -> dict:
        """从请求的多模态输入中收集推理所需路径。

        v0.6+：从 engine.py 的 InferRequest 中提取图片/视频路径，
        按 mode 分配到 wrapper.generate() 的对应参数。
        v0.7+：ref2va 多模态参考模式——收集图片/视频/音频三类参考路径，
        由 wrapper 构造混合 references 列表（MiniMaxH3ImageReference /
        VideoReference / AudioReference）。

        Args:
            mode: 生成模式（t2v / i2v / fl2v / ref2v / ref2va / video_edit）
            images: InferRequest.images 列表（每项含 path/url）
            videos: InferRequest.videos 列表
            audios: InferRequest.audios 列表（ref2va 模式音频参考）

        Returns:
            推理参数字典（image / last_image / references / video /
            reference_images / reference_videos / reference_audios 键）
        """
        result: dict[str, Any] = {}

        if not images:
            images = []
        if not videos:
            videos = []
        if not audios:
            audios = []

        # 取第一张图片作为首帧 / 参考图（I2V / FL2V / Ref2VA 模式）
        if mode in ("i2v", "fl2v", "ref2v") and images:
            first_img = images[0]
            image_input = first_img.path or first_img.url
            if image_input:
                result["image"] = image_input

            # FL2V 模式：第二张图片作为尾帧
            if mode == "fl2v" and len(images) >= 2:
                last_img = images[1]
                last_image_input = last_img.path or last_img.url
                if last_image_input:
                    result["last_image"] = last_image_input

        # Ref2V 模式：收集所有参考图片（最多 9 张）
        if mode == "ref2v" and images:
            references = [
                img.path or img.url
                for img in images
                if img.path or img.url
            ]
            if references:
                result["references"] = references[:9]

        # Ref2VA 多模态参考模式：收集图片/视频/音频三类参考路径
        # （engine.py 已做入口校验：音频必须配对图/视频、数量上限 9/3/3）
        if mode == "ref2va":
            ref_images = [
                img.path or img.url
                for img in images
                if img.path or img.url
            ]
            ref_videos = [
                vid.path or vid.url
                for vid in videos
                if vid.path or vid.url
            ]
            ref_audios = [
                aud.path or aud.url
                for aud in audios
                if aud.path or aud.url
            ]
            if ref_images:
                result["reference_images"] = ref_images[:9]
            if ref_videos:
                result["reference_videos"] = ref_videos[:3]
            if ref_audios:
                result["reference_audios"] = ref_audios[:3]

        # 取第一个视频作为参考视频（video_edit 模式）
        if mode == "video_edit" and videos:
            first_vid = videos[0]
            video_input = first_vid.path or first_vid.url
            if video_input:
                result["video"] = video_input

        self._logger.info(
            "多模态输入收集: mode=%s, has_image=%s, has_last_image=%s, has_references=%s, "
            "has_video=%s, ref_images=%d, ref_videos=%d, ref_audios=%d",
            mode,
            "image" in result,
            "last_image" in result,
            "references" in result,
            "video" in result,
            len(result.get("reference_images", [])),
            len(result.get("reference_videos", [])),
            len(result.get("reference_audios", [])),
        )
        return result
