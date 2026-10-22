"""MiniMax-H3 视频生成引擎 Gradio Web UI。

页面元素：
    - 状态栏（引擎状态 / 已加载模型 / 显存 / offload 等级）
    - 模型选择 + 加载/卸载/刷新按钮
    - CPU Offload 等级选择（L0/L1/L2/L3/auto）
    - v0.6+ 加载方案管理（A/B/C/D 多档显存，保存为默认）
    - v0.6+ 高级模式（手动开关加速方案：SageAttention / LightX2V LoRA / FirstBlockCache / BlockSwap）
    - 5 模式 Tab：
        - T2V  文生视频
        - I2V  图生视频（首帧/尾帧）
        - FL2V 首尾帧生视频
        - Ref2V 多宫格生视频（≤9 图）
        - VideoEdit 视频编辑（参考视频 + prompt）
    - LoRA 管理（加载/卸载/查看已加载）
    - 高级参数（duration / fps / 分辨率 / steps / guidance / seed）
    - 生成按钮 + 进度状态（SSE 流式）
    - 视频播放器 + 音频播放器 + 下载链接
    - 日志显示

设计原则：
    - 调用自身 API（不直接调 engine.py），保证 UI 和 API 行为一致
    - SSE 流式消费进度帧（video.start → video.completed）
    - 自动加载逻辑（未加载时自动 /load）
    - 支持 iframe 嵌入（root_path 由 server.py 配置）
    - 多 LoRA 管理通过 /lora/load + /lora/unload 端点
    - CPU Offload 等级通过 LoadRequest.inputs.cpu_offload_level 透传
    - v0.6+ 方案管理通过 /schemes + /schemes/default 端点
    - v0.6+ 高级模式通过 inputs.acceleration_override 传递加速方案覆盖
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import httpx

logger = logging.getLogger(__name__)


# ===== 输出目录根（与 config.EngineConfig.output_root 一致）=====
# 用于将 /output/video/xxx.mp4 形式的 URL 映射到本地文件路径，
# 绕过 Gradio safehttpx 对 127.0.0.1 的安全拦截（"Hostname 127.0.0.1 failed validation"）
# 进程 CWD 为项目根目录，所以相对路径 "output" 即指向 {project_root}/output
_OUTPUT_ROOT = Path("output")


def _url_to_local_path(url: str) -> Optional[str]:
    """将引擎返回的 /output/... URL 映射到本地文件绝对路径。

    Gradio 的 safehttpx 会拦截 127.0.0.1 这种本地 IP，导致 gr.Video / gr.Audio
    收到 http://127.0.0.1:port/output/... URL 时抛
    "Hostname 127.0.0.1 failed validation" 错误。

    解决方案：URL 形如 /output/video/xxx.mp4 → 本地 output/video/xxx.mp4
    若文件存在则返回绝对路径（Gradio 会用本地文件，不触发 safehttpx）。

    Args:
        url: 引擎返回的 URL 路径，如 /output/video/xxx.mp4

    Returns:
        本地文件绝对路径字符串；文件不存在时返回 None（调用方回退到 URL）
    """
    if not url:
        return None
    # 去掉开头的 / 让它变成相对路径
    rel = url.lstrip("/")
    # 只接受 output/ 前缀（防止路径穿越）
    if not rel.startswith("output/"):
        return None
    local_path = _OUTPUT_ROOT.parent / rel if _OUTPUT_ROOT.is_absolute() else Path(rel)
    if local_path.exists():
        return str(local_path.resolve())
    return None


# ===== 模式标签 → 内部 task_type 映射 =====
# 显示用中文标签，API 调用用英文 task_type
MODE_T2V = "t2v"                    # 文生视频
MODE_I2V = "i2v"                    # 图生视频
MODE_FL2V = "fl2v"                  # 首尾帧生视频
MODE_REF2V = "ref2v"                # 多宫格生视频
MODE_VIDEO_EDIT = "video_edit"      # 视频编辑

# CPU offload 等级选项（与 schemas.OFFLOAD_LEVELS 一致）
OFFLOAD_CHOICES = ["auto", "L0", "L1", "L2", "L3"]

OFFLOAD_DESCRIPTIONS = {
    "auto": "自动（根据 GPU 显存选择，推荐）",
    "L0": "L0 - 全 GPU（≥80GB 显存，最快）",
    "L1": "L1 - Text Encoder 卸载（≥48GB 显存）",
    "L2": "L2 - 层级别 offload（≥24GB 显存，慢 3-5×）",
    "L3": "L3 - 激进 offload（<24GB 显存，慢 10×+）",
}


def _parse_variant_label(variant_label: str) -> str:
    """从显示 label 解析出实际 variant name（用于 /load API 调用）。

    v0.5+ diffusers 格式：label 格式为
        "[FL2VA] MiniMax/H3-FL2VA (768x432, bf16, 含音频)"
    原写法 `label.split(" ")[0]` 在新格式下会拿到 "[FL2VA]"，导致 /load 失败。
    此处用正则提取形如 `MiniMax/H3-xxx` 的 variant name。

    Args:
        variant_label: 下拉框选中项的显示文本

    Returns:
        实际 variant name（如 "MiniMax/H3-FL2VA"），
        解析失败时回退到 split(" ")[0]（向后兼容旧格式）
    """
    import re
    if not variant_label:
        return ""
    # 提取 MiniMax/H3-... 格式的 variant name
    m = re.search(r"(MiniMax/H3-[A-Za-z0-9_\-]+)", variant_label)
    if m:
        return m.group(1)
    # 回退：去掉可能的 [...] 前缀后取第二段
    parts = variant_label.split(" ")
    if len(parts) >= 2 and parts[0].startswith("["):
        return parts[1]
    return parts[0] if parts else ""


def create_ui(engine_url: str, theme: Optional[gr.Theme] = None) -> gr.Blocks:
    """创建 MiniMax-H3 视频生成引擎 Web UI。

    Args:
        engine_url: 引擎 HTTP 地址（如 http://127.0.0.1:7006）
        theme: Gradio 主题（Gradio 6.0+ 不再传给 Blocks，由 mount_gradio_app 应用）

    Returns:
        gr.Blocks: UI 实例
    """
    with gr.Blocks(title="MiniMax-H3 视频生成") as demo:
        gr.Markdown("# MiniMax-H3 视频生成引擎")
        gr.Markdown(
            "支持 5 种生成模式：文生视频 / 图生视频 / 首尾帧 / 多宫格 / 视频编辑 "
            "（FL2VA + Ref2VA 双范式，含原生音频）"
        )

        # ===== v0.6.3+ 弹窗浮层 CSS（Gradio 6.22 无原生 Modal，用 CSS+JS 模拟）=====
        # 外层 Column（.h3-modal）默认隐藏，添加 .open class 后变为 fixed 浮层 + 遮罩
        # 内层 Column（.h3-modal-card）为白色居中卡片
        gr.HTML(
            """
<style>
/* 外层遮罩层：默认隐藏 */
.h3-modal {
    display: none !important;
}
/* .open 时显示为全屏 fixed 浮层 + 半透明遮罩 */
.h3-modal.open {
    display: flex !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(0, 0, 0, 0.5) !important;
    z-index: 99999 !important;
    justify-content: center !important;
    align-items: flex-start !important;
    padding-top: 8vh !important;
    overflow-y: auto !important;
}
/* 内层白色卡片 */
.h3-modal-card {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 24px !important;
    max-width: 680px !important;
    width: 90% !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 8vh !important;
}
</style>
            """
        )

        # ===== v0.6.1+ 方案联动 State（存储 API 返回数据，供方案变化时过滤 variant）=====
        caps_state = gr.State({})      # /capabilities 返回的 variants 列表
        schemes_state = gr.State({})   # /schemes 返回的 schemes 列表（含 applicable_variants）

        # ================================================================
        # ① 配置区
        # ================================================================
        gr.Markdown("### ① 配置区")

        # 状态栏精简：1行HTML 展示 state/模型/显存/offload
        status_display = gr.HTML(
            value="<div style='padding:8px 12px; background:#f0f4f8; border-radius:6px; "
                  "font-size:14px;'>● 未连接</div>",
        )
        # 隐藏的 Textbox 供 refresh_all 更新（HTML 无法直接被 Textbox 赋值，
        # 用 HTML 组件直接接收 HTML 字符串）
        variant_display = gr.Textbox(visible=False)
        vram_display = gr.Textbox(visible=False)
        offload_display = gr.Textbox(visible=False)

        # 配置区按钮行
        with gr.Row():
            scheme_dialog_btn = gr.Button("⚙ 方案管理", variant="secondary", scale=1)
            adv_dialog_btn = gr.Button("⚙ 高级设置", variant="secondary", scale=1)
            refresh_btn = gr.Button("刷新状态", variant="secondary", scale=1)

        # 模型选择 + offload + 加载/卸载
        with gr.Row():
            variant_select = gr.Dropdown(
                label="模型变体（Variant）",
                choices=[],
                value=None,
                interactive=True,
                info="由方案+功能Tab自动过滤：方案决定精度(SDNQ/BF16) | Tab决定范式(FL2VA/Ref2VA)",
                scale=2,
            )
            offload_select = gr.Dropdown(
                label="CPU Offload 等级",
                choices=OFFLOAD_CHOICES,
                value="auto",
                interactive=True,
                info="由方案决定（A→L2 / B→L2 / C→L1 / D→L0），可手动微调",
                scale=1,
            )
            load_btn = gr.Button("加载模型", variant="primary", scale=1)
            unload_btn = gr.Button("卸载模型", variant="stop", scale=1)

        # ================================================================
        # 弹窗1：方案管理（CSS 浮层 + JS 控制 class，v0.6.3+ 真正的模态弹窗）
        # 外层 Column（#scheme-modal）作为遮罩层，内层 .h3-modal-card 为白色卡片
        # ================================================================
        scheme_dialog = gr.Column(
            elem_id="scheme-modal", elem_classes=["h3-modal"], variant="compact"
        )
        with scheme_dialog:
            with gr.Column(elem_classes=["h3-modal-card"]):
                gr.Markdown("## ⚙ 加载方案管理")
                gr.Markdown(
                    "<div style='padding:8px 12px; background:#fff3cd; border-radius:6px; "
                    "border:1px solid #ffeaa7; margin-bottom:8px;'>"
                    "选择方案后自动联动：模型变体过滤 / offload等级 / 推理步数 / 加速方案"
                    "</div>"
                )
                with gr.Row():
                    gpu_vram_display = gr.Textbox(
                        label="检测到的 GPU 显存",
                        interactive=False,
                        value="-",
                        scale=1,
                    )
                    recommended_scheme = gr.Textbox(
                        label="推荐方案",
                        interactive=False,
                        value="-",
                        scale=1,
                    )
                scheme_select = gr.Dropdown(
                    label="加载方案",
                    choices=[],
                    value=None,
                    interactive=True,
                    info="A/B/C→SDNQ 4bit 量化模型 | D→BF16 原版模型",
                )
                scheme_desc = gr.Textbox(
                    label="方案描述",
                    interactive=False,
                    value="-",
                    lines=4,
                )
                scheme_status = gr.Textbox(
                    label="方案操作结果",
                    interactive=False,
                    value="-",
                    lines=1,
                )
                with gr.Row():
                    scheme_save_btn = gr.Button("保存为默认", variant="primary", scale=1)
                    scheme_refresh_btn = gr.Button("刷新方案", variant="secondary", scale=1)
                    scheme_close_btn = gr.Button("关闭", variant="stop", scale=1)

        # ================================================================
        # 弹窗2：高级设置（CSS 浮层 + JS 控制 class，Tab 分组：加速方案 / 高级参数 / LoRA管理）
        # ================================================================
        adv_dialog = gr.Column(
            elem_id="adv-modal", elem_classes=["h3-modal"], variant="compact"
        )
        with adv_dialog:
            with gr.Column(elem_classes=["h3-modal-card"]):
                gr.Markdown("## ⚙ 高级设置")
                with gr.Tabs():
                    # --- Tab 1: 加速方案 ---
                    with gr.Tab("加速方案"):
                        gr.Markdown(
                            "**说明**：加速方案在模型加载时注入，修改后需重新加载模型生效。\n"
                            "- SageAttention：INT8 QK + FP8 PV 注意力量化，加速 1.5-2×\n"
                            "- LightX2V Turbo LoRA：4/8-step 蒸馏 LoRA，加速 6-12×\n"
                            "- FirstBlockCache：首块必跑 + 残差复用，加速 1.5×\n"
                            "- BlockSwap：transformer 块卸载到 CPU（方案 C 用）"
                        )
                        with gr.Row():
                            adv_sage = gr.Checkbox(
                                label="SageAttention",
                                value=False,
                                interactive=True,
                            )
                            adv_lora = gr.Checkbox(
                                label="LightX2V Turbo LoRA",
                                value=False,
                                interactive=True,
                            )
                            adv_fbc = gr.Checkbox(
                                label="FirstBlockCache",
                                value=False,
                                interactive=True,
                            )
                            adv_blockswap = gr.Checkbox(
                                label="BlockSwap",
                                value=False,
                                interactive=True,
                            )
                        with gr.Row():
                            adv_lora_scale = gr.Slider(
                                label="LoRA 权重（scale）",
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                interactive=True,
                                scale=2,
                            )
                            adv_apply_btn = gr.Button(
                                "应用并重新加载模型",
                                variant="primary",
                                scale=2,
                            )
                            adv_status = gr.Textbox(
                                label="状态",
                                interactive=False,
                                value="-",
                                scale=3,
                            )

                    # --- Tab 2: 高级参数 ---
                    with gr.Tab("高级参数"):
                        with gr.Row():
                            guidance_slider = gr.Slider(
                                label="CFG 引导强度（guidance_scale）",
                                minimum=0.0,
                                maximum=20.0,
                                value=7.0,
                                step=0.5,
                            )
                            seed_input = gr.Number(
                                label="随机种子",
                                value=-1,
                                info="-1 = 随机种子，固定种子可复现结果",
                            )

                    # --- Tab 3: LoRA 管理 ---
                    with gr.Tab("LoRA 管理"):
                        with gr.Row():
                            lora_name = gr.Dropdown(
                                label="LoRA 名称",
                                choices=[],
                                interactive=True,
                                scale=2,
                            )
                            lora_weight = gr.Slider(
                                label="LoRA 权重",
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                scale=1,
                            )
                        with gr.Row():
                            lora_load_btn = gr.Button("加载 LoRA", variant="primary", scale=1)
                            lora_unload_btn = gr.Button("卸载 LoRA", variant="stop", scale=1)
                        lora_status = gr.Textbox(
                            label="LoRA 操作结果",
                            interactive=False,
                            value="-",
                        )

                with gr.Row():
                    adv_close_btn = gr.Button("关闭", variant="stop", scale=1)

        # ===== 5 模式 Tab =====
        with gr.Tabs() as tabs:
            # ----- Tab 1: T2V 文生视频 -----
            with gr.Tab("T2V 文生视频") as tab_t2v:
                gr.Markdown(
                    "**文生视频**：仅根据文本描述生成视频（含原生音频）。"
                    "使用 FL2VA variant + 0 图输入。"
                )
                t2v_prompt = gr.Textbox(
                    label="正向提示词（Prompt）",
                    placeholder="如：一只金毛犬在海边奔跑，夕阳西下，电影级运镜...",
                    lines=3,
                )
                t2v_negative = gr.Textbox(
                    label="负向提示词（可选）",
                    placeholder="如：模糊、低质量、变形、水印...",
                    lines=1,
                )

            # ----- Tab 2: I2V 图生视频 -----
            with gr.Tab("I2V 图生视频") as tab_i2v:
                gr.Markdown(
                    "**图生视频**：上传 1 张图片作为首帧或尾帧，根据文本生成视频。"
                    "使用 FL2VA variant + 1 图输入。"
                )
                i2v_image = gr.Image(
                    label="输入图片（作为首帧或尾帧）",
                    type="filepath",
                    sources=["upload"],
                )
                i2v_role = gr.Radio(
                    label="图片角色",
                    choices=["first_frame（首帧）", "last_frame（尾帧）"],
                    value="first_frame（首帧）",
                    info="首帧：图片作为视频开头；尾帧：图片作为视频结尾",
                )
                i2v_prompt = gr.Textbox(
                    label="正向提示词",
                    placeholder="如：镜头缓慢推进，人物微笑...",
                    lines=3,
                )
                i2v_negative = gr.Textbox(
                    label="负向提示词（可选）",
                    lines=1,
                )

            # ----- Tab 3: FL2V 首尾帧生视频 -----
            with gr.Tab("FL2V 首尾帧") as tab_fl2v:
                gr.Markdown(
                    "**首尾帧生视频**：上传首帧和尾帧两张图片，模型在两帧之间生成过渡视频。"
                    "使用 FL2VA variant + 2 图输入。"
                )
                with gr.Row():
                    fl2v_first = gr.Image(
                        label="首帧图片",
                        type="filepath",
                        sources=["upload"],
                    )
                    fl2v_last = gr.Image(
                        label="尾帧图片",
                        type="filepath",
                        sources=["upload"],
                    )
                fl2v_prompt = gr.Textbox(
                    label="正向提示词（描述两帧之间的过渡）",
                    placeholder="如：镜头平滑推进，云朵飘动...",
                    lines=3,
                )
                fl2v_negative = gr.Textbox(
                    label="负向提示词（可选）",
                    lines=1,
                )

            # ----- Tab 4: Ref2V 多宫格生视频 -----
            with gr.Tab("Ref2V 多宫格") as tab_ref2v:
                gr.Markdown(
                    "**多宫格生视频**：上传最多 9 张参考图片，模型根据多图引用生成视频。"
                    "使用 Ref2VA variant + ≤9 图输入。"
                )
                ref2v_gallery = gr.Gallery(
                    label="参考图片（最多 9 张，按上传顺序作为 <Subject 1> / <Subject 2> / ...）",
                    type="filepath",
                    columns=3,
                    height=300,
                    sources=["upload"],
                )
                ref2v_prompt = gr.Textbox(
                    label="正向提示词（描述如何组合参考图）",
                    placeholder="如：<Subject 1> 走向 <Subject 2>，背景为森林...",
                    lines=3,
                    info="提示词中可用 <Subject N> 引用第 N 张图片",
                )
                ref2v_negative = gr.Textbox(
                    label="负向提示词（可选）",
                    lines=1,
                )

            # ----- Tab 5: 视频编辑 -----
            with gr.Tab("视频编辑") as tab_edit:
                gr.Markdown(
                    "**视频编辑**：上传源视频 + 文本指令，模型对视频进行编辑。"
                    "使用 Ref2VA variant + 1 视频输入。"
                )
                edit_video = gr.Video(
                    label="源视频（待编辑）",
                    sources=["upload"],
                )
                edit_prompt = gr.Textbox(
                    label="编辑指令（Prompt）",
                    placeholder="如：将视频中的白天改为夜晚，添加霓虹灯效果...",
                    lines=3,
                    info="支持 <Video N> / <Audio N> 标签引用源视频内容",
                )
                edit_negative = gr.Textbox(
                    label="负向提示词（可选）",
                    lines=1,
                )

        # ================================================================
        # ② 参数选择区（常用参数内联）
        # ================================================================
        gr.Markdown("### ② 参数选择区")
        with gr.Row():
            duration_slider = gr.Slider(
                label="视频时长（秒）",
                minimum=1,
                maximum=15,
                value=5,
                step=1,
                info="MiniMax-H3 最长支持 15 秒",
                scale=1,
            )
            fps_slider = gr.Slider(
                label="帧率（FPS）",
                minimum=8,
                maximum=30,
                value=24,
                step=1,
                scale=1,
            )
            resolution_select = gr.Dropdown(
                label="分辨率",
                choices=[
                    "480x256 (16:9 快速测试)",
                    "256x480 (9:16 快速测试)",
                    "768x432 (16:9)",
                    "432x768 (9:16)",
                    "576x576 (1:1)",
                    "1024x576 (16:9 高清)",
                    "576x1024 (9:16 高清)",
                ],
                value="768x432 (16:9)",
                info="宽 x 高，需为 8 的倍数；小尺寸用于功能测试，大尺寸用于正式产出",
                scale=1,
            )
            steps_slider = gr.Slider(
                label="推理步数（steps）",
                minimum=1,
                maximum=200,
                value=50,
                step=1,
                info="方案A/B/C→4/8步 | 方案D→50步",
                scale=1,
            )

        # ================================================================
        # ③ 生成按钮 + 进度
        # ================================================================
        gr.Markdown("### ③ 生成")
        with gr.Row():
            generate_btn = gr.Button(
                "✨ 生成视频", variant="primary", scale=2
            )
            progress_display = gr.Textbox(
                label="进度",
                interactive=False,
                value="就绪",
                scale=3,
            )

        # ================================================================
        # ④ 结果展示区
        # ================================================================
        gr.Markdown("### ④ 结果展示")
        with gr.Row():
            video_output = gr.Video(
                label="生成的视频（含音频）",
                interactive=False,
                scale=2,
            )
            audio_output = gr.Audio(
                label="生成的音频（独立音轨）",
                interactive=False,
                scale=1,
            )
        with gr.Row():
            video_download = gr.Textbox(
                label="视频下载链接",
                interactive=False,
                value="-",
                scale=1,
            )
            audio_download = gr.Textbox(
                label="音频下载链接",
                interactive=False,
                value="-",
                scale=1,
            )
            meta_display = gr.Textbox(
                label="生成元信息",
                interactive=False,
                value="-",
                scale=2,
            )

        # ===== 日志（折叠） =====
        with gr.Accordion("日志输出", open=False):
            log_display = gr.Textbox(
                label="日志",
                interactive=False,
                lines=5,
                max_lines=10,
            )

        # ===== 事件绑定 =====
        current_tab = gr.State(MODE_T2V)

        # v0.6.2+ Tab 切换时更新 current_tab + 范式过滤 variant + 模型不匹配检查
        def _on_tab_change(tab_id, scheme_label, caps_data, schemes_data):
            return on_tab_change_filter(
                engine_url, tab_id, scheme_label, caps_data, schemes_data
            )

        tab_t2v.select(
            fn=lambda *a: _on_tab_change(MODE_T2V, *a),
            inputs=[scheme_select, caps_state, schemes_state],
            outputs=[variant_select, current_tab],
        )
        tab_i2v.select(
            fn=lambda *a: _on_tab_change(MODE_I2V, *a),
            inputs=[scheme_select, caps_state, schemes_state],
            outputs=[variant_select, current_tab],
        )
        tab_fl2v.select(
            fn=lambda *a: _on_tab_change(MODE_FL2V, *a),
            inputs=[scheme_select, caps_state, schemes_state],
            outputs=[variant_select, current_tab],
        )
        tab_ref2v.select(
            fn=lambda *a: _on_tab_change(MODE_REF2V, *a),
            inputs=[scheme_select, caps_state, schemes_state],
            outputs=[variant_select, current_tab],
        )
        tab_edit.select(
            fn=lambda *a: _on_tab_change(MODE_VIDEO_EDIT, *a),
            inputs=[scheme_select, caps_state, schemes_state],
            outputs=[variant_select, current_tab],
        )

        # ===== 弹窗按钮事件（v0.6.3+ 用 JS 控制 class 实现浮层显示/隐藏）=====
        # 不再使用 gr.update(visible=...)（那只是内联展开/折叠），改用 JS 切换 .open class
        scheme_dialog_btn.click(
            fn=None,
            js="(e) => { const el = document.getElementById('scheme-modal'); if (el) el.classList.add('open'); }",
        )
        scheme_close_btn.click(
            fn=None,
            js="(e) => { const el = document.getElementById('scheme-modal'); if (el) el.classList.remove('open'); }",
        )
        adv_dialog_btn.click(
            fn=None,
            js="(e) => { const el = document.getElementById('adv-modal'); if (el) el.classList.add('open'); }",
        )
        adv_close_btn.click(
            fn=None,
            js="(e) => { const el = document.getElementById('adv-modal'); if (el) el.classList.remove('open'); }",
        )

        # 页面加载 + 刷新按钮
        def _on_refresh():
            return refresh_all(engine_url)

        _refresh_outputs = [
            status_display, variant_display, vram_display, offload_display,
            variant_select,
            scheme_select, scheme_desc, gpu_vram_display,
            recommended_scheme, scheme_status,
            adv_sage, adv_lora, adv_fbc, adv_blockswap, adv_lora_scale, adv_status,
            caps_state, schemes_state,
        ]

        demo.load(fn=_on_refresh, outputs=_refresh_outputs)
        refresh_btn.click(fn=_on_refresh, outputs=_refresh_outputs)

        # 加载按钮
        def _on_load(variant, offload_level):
            return load_variant(engine_url, variant, offload_level)

        load_btn.click(
            fn=_on_load,
            inputs=[variant_select, offload_select],
            outputs=[status_display, variant_display, vram_display, offload_display, log_display],
        )

        # 卸载按钮
        def _on_unload():
            return unload_variant(engine_url)

        unload_btn.click(
            fn=_on_unload,
            outputs=[status_display, variant_display, vram_display, offload_display, log_display],
        )

        # ===== 方案管理事件 =====

        def _on_scheme_change(scheme_label, tab_id, caps_data, schemes_data):
            return on_scheme_change_linkage(
                engine_url, scheme_label, tab_id, caps_data, schemes_data
            )

        scheme_select.change(
            fn=_on_scheme_change,
            inputs=[scheme_select, current_tab, caps_state, schemes_state],
            outputs=[
                scheme_desc,
                variant_select, offload_select, steps_slider,
                adv_sage, adv_lora, adv_fbc, adv_blockswap, adv_lora_scale, adv_status,
            ],
        )

        def _on_scheme_save(scheme_id):
            return save_default_scheme(engine_url, scheme_id)

        scheme_save_btn.click(
            fn=_on_scheme_save,
            inputs=[scheme_select],
            outputs=[scheme_status, log_display],
        )

        def _on_scheme_refresh():
            return refresh_schemes(engine_url)

        scheme_refresh_btn.click(
            fn=_on_scheme_refresh,
            outputs=[
                scheme_select, scheme_desc, gpu_vram_display,
                recommended_scheme, scheme_status,
            ],
        )

        # ===== v0.6+ 高级模式事件 =====

        # 应用并重新加载模型（修改加速方案后触发 unload + load）
        def _on_adv_apply(
            variant, offload_level,
            adv_sage_v, adv_lora_v, adv_fbc_v, adv_blockswap_v, adv_lora_scale_v,
        ):
            yield from apply_acceleration_override(
                engine_url, variant, offload_level,
                adv_sage_v, adv_lora_v, adv_fbc_v, adv_blockswap_v, adv_lora_scale_v,
            )

        adv_apply_btn.click(
            fn=_on_adv_apply,
            inputs=[
                variant_select, offload_select,
                adv_sage, adv_lora, adv_fbc, adv_blockswap, adv_lora_scale,
            ],
            outputs=[adv_status, status_display, variant_display, vram_display, offload_display, log_display],
        )

        # LoRA 加载
        def _on_lora_load(name, weight):
            return lora_load(engine_url, name, weight)

        lora_load_btn.click(
            fn=_on_lora_load,
            inputs=[lora_name, lora_weight],
            outputs=[lora_status, log_display],
        )

        # LoRA 卸载
        def _on_lora_unload(name):
            return lora_unload(engine_url, name)

        lora_unload_btn.click(
            fn=_on_lora_unload,
            inputs=[lora_name],
            outputs=[lora_status, log_display],
        )

        # 生成按钮（根据当前 Tab 决定参数）
        def _on_generate(
            tab, variant, offload_level,
            t2v_p, t2v_n,
            i2v_img, i2v_role, i2v_p, i2v_n,
            fl2v_first, fl2v_last, fl2v_p, fl2v_n,
            ref2v_imgs, ref2v_p, ref2v_n,
            edit_vid, edit_p, edit_n,
            duration, fps, resolution, steps, guidance, seed,
        ):
            yield from generate_video(
                engine_url, tab, variant, offload_level,
                t2v_p, t2v_n,
                i2v_img, i2v_role, i2v_p, i2v_n,
                fl2v_first, fl2v_last, fl2v_p, fl2v_n,
                ref2v_imgs, ref2v_p, ref2v_n,
                edit_vid, edit_p, edit_n,
                duration, fps, resolution, steps, guidance, seed,
            )

        generate_btn.click(
            fn=_on_generate,
            inputs=[
                current_tab, variant_select, offload_select,
                t2v_prompt, t2v_negative,
                i2v_image, i2v_role, i2v_prompt, i2v_negative,
                fl2v_first, fl2v_last, fl2v_prompt, fl2v_negative,
                ref2v_gallery, ref2v_prompt, ref2v_negative,
                edit_video, edit_prompt, edit_negative,
                duration_slider, fps_slider, resolution_select,
                steps_slider, guidance_slider, seed_input,
            ],
            outputs=[
                progress_display, video_output, audio_output,
                video_download, audio_download, meta_display, log_display,
            ],
        )

    return demo


# ===== API 调用函数 =====

def refresh_all(engine_url: str) -> tuple:
    """刷新状态、variant 列表 + v0.6+ 方案管理区 + 高级模式状态。

    返回 15 个字段（按顺序）：
        status_text, variant_display, vram_display, offload_display, variant_select_update,
        scheme_select_update, scheme_desc, gpu_vram_display, recommended_scheme, scheme_status,
        adv_sage, adv_lora, adv_fbc, adv_blockswap, adv_lora_scale, adv_status
    """
    try:
        status = _http_get(engine_url, "/status")
        caps = _http_get(engine_url, "/capabilities")

        status_text = f"state={status.get('state', '?')}"
        extra = status.get("extra", {})
        device_display = extra.get("device_name") or extra.get("device")
        if device_display:
            status_text += f" | device={device_display}"
        if extra.get("is_loaded"):
            status_text += " | 已加载"

        variant_display = status.get("loaded_variant") or "-"
        vram_display = status.get("vram_used", "-")
        if isinstance(vram_display, (int, float)):
            vram_display = f"{vram_display:.0f}MB"
        offload_display = extra.get("offload_level", "-")

        # variant 列表（v0.4.x+ 改造）：
        # 1. 只显示已下载的（available=True），未下载的不列在 UI（用户去详情页下载）
        # 2. 按 base_model（FL2VA / Ref2VA）加功能前缀，方便区分两种范式
        # 3. 显示分辨率 + 音频支持
        # 4. 列表为空时给出占位项（避免下拉框空导致 disabled 状态）
        variants = []
        for v in caps.get("variants", []):
            # v0.4.x：只列已下载的 variant
            if not v.get("available", False):
                continue
            name = v.get("name", "")
            base_model = v.get("base_model", "")            # FL2VA / Ref2VA
            precision = v.get("precision", "")              # bf16 / int8
            # v0.4.x：从 model_scanner 的 _KNOWN_VARIANTS 读取 resolution，
            # 兜底改为 "未知" 避免显示生硬的 "?"
            resolution = v.get("resolution") or "未知"
            supports_audio = v.get("supports_audio", False)
            audio_tag = "含音频" if supports_audio else "无音频"
            # label 格式：[FL2VA] MiniMax/H3-FL2VA (768x432, bf16, 含音频)
            func_tag = f"[{base_model}]" if base_model else ""
            label = f"{func_tag} {name} ({resolution}, {precision}, {audio_tag})".strip()
            variants.append(label)

        # v0.4.x：无可用 variant 时显示占位，避免空下拉框
        if not variants:
            variants = ["暂无可用模型，请到引擎详情页下载"]
            default = None
        else:
            # Gradio Dropdown value 需精确匹配 choices 某项，
            # 所以 default 取完整 label（与 /load 时的解析解耦）
            default = variants[0]

        variant_select_update = gr.update(choices=variants, value=default)

        # ===== v0.6+ 方案管理区刷新 =====
        scheme_select_update, scheme_desc_v, gpu_vram_v, rec_scheme_v, scheme_status_v = (
            _build_scheme_panel(engine_url, status)
        )

        # ===== v0.6+ 高级模式加速方案状态刷新（显示当前已加载方案的加速配置）=====
        adv_sage_v, adv_lora_v, adv_fbc_v, adv_blockswap_v, adv_lora_scale_v, adv_status_v = (
            _build_advanced_panel(engine_url, status)
        )

        # v0.6.1+ 获取 schemes 数据（含 applicable_variants，供方案联动使用）
        try:
            schemes_data = _http_get(engine_url, "/schemes")
        except Exception:
            schemes_data = {}

        return (
            status_text, variant_display, vram_display, offload_display, variant_select_update,
            scheme_select_update, scheme_desc_v, gpu_vram_v, rec_scheme_v, scheme_status_v,
            adv_sage_v, adv_lora_v, adv_fbc_v, adv_blockswap_v, adv_lora_scale_v, adv_status_v,
            caps, schemes_data,  # v0.6.1+ 方案联动 State
        )
    except Exception as e:
        logger.error("刷新失败: %s", e)
        err_msg = f"连接失败: {e}"
        return (
            err_msg, "-", "-", "-", gr.update(choices=[]),
            gr.update(choices=[], value=None), "-", "-", "-", "-",
            False, False, False, False, 1.0, "-",
            {}, {},  # v0.6.1+ 方案联动 State（空）
        )


def refresh_schemes(engine_url: str) -> tuple:
    """v0.6+：单独刷新方案管理区（不刷新 variant 列表）。

    返回 5 个字段：
        scheme_select_update, scheme_desc, gpu_vram_display, recommended_scheme, scheme_status
    """
    try:
        status = _http_get(engine_url, "/status")
        return _build_scheme_panel(engine_url, status)
    except Exception as e:
        logger.error("刷新方案列表失败: %s", e)
        return gr.update(choices=[], value=None), "-", "-", "-", f"刷新失败: {e}"


def _build_scheme_panel(engine_url: str, status: dict) -> tuple:
    """构建方案管理区的 UI 数据。

    Args:
        engine_url: 引擎 URL
        status: /status 返回的 dict

    Returns:
        (scheme_select_update, scheme_desc, gpu_vram_display, recommended_scheme, scheme_status)
    """
    try:
        schemes_resp = _http_get(engine_url, "/schemes")
    except Exception as e:
        logger.error("获取方案列表失败: %s", e)
        return (
            gr.update(choices=[], value=None),
            f"获取方案列表失败: {e}",
            "-", "-", "-",
        )

    schemes = schemes_resp.get("schemes", [])
    default_scheme_id = schemes_resp.get("default_scheme_id")
    loaded_scheme_id = schemes_resp.get("loaded_scheme_id")

    # 构造下拉框 choices：每项格式 "A - 24G 显存流畅版（SDNQ 4bit）"
    scheme_choices = []
    scheme_desc_map: dict[str, str] = {}
    for s in schemes:
        sid = s.get("id", "?")
        sname = s.get("name", "")
        sdesc = s.get("description", "")
        sprecision = s.get("precision", "")
        svram = s.get("vram_required", "")
        soffload = s.get("offload_level", "")
        label = f"{sid} - {sname}（{sprecision}）"
        scheme_choices.append(label)
        scheme_desc_map[sid] = (
            f"{sname}\n"
            f"描述：{sdesc}\n"
            f"精度：{sprecision} | 显存需求：{svram} | offload：{soffload}"
        )

    # 默认选中：当前已加载方案 > 默认方案 > 第一个
    selected_sid = loaded_scheme_id or default_scheme_id
    selected_label = None
    if selected_sid:
        for lbl in scheme_choices:
            if lbl.startswith(f"{selected_sid} - "):
                selected_label = lbl
                break
    if not selected_label and scheme_choices:
        selected_label = scheme_choices[0]

    scheme_select_update = gr.update(choices=scheme_choices, value=selected_label)

    # 描述
    scheme_desc_v = scheme_desc_map.get(selected_sid, "-") if selected_sid else "-"

    # 检测到的 GPU 显存（从 status.extra.available_schemes 推断，或显示 "-"）
    extra = status.get("extra", {})
    available_schemes = extra.get("available_schemes", [])

    # 通过 /schemes 接口无法获取 GPU 显存，这里用 status.extra 中的信息推断
    # 简化：显示 "查看推荐方案" 区域
    gpu_vram_v = "（请查看推荐方案）"

    # 推荐方案：从 available_schemes 中按显存自动选档（这里简化，显示 default_scheme_id）
    # 真正的自动选档逻辑在 scheme_manager.auto_select_scheme 中
    # 由于 /status 不返回 GPU 显存，这里只显示当前默认方案
    if default_scheme_id:
        # 找到默认方案的 name
        default_name = next(
            (s.get("name", "") for s in schemes if s.get("id") == default_scheme_id),
            "",
        )
        rec_scheme_v = f"{default_scheme_id} - {default_name}"
    else:
        rec_scheme_v = "未设置（启动时自动选档）"

    # 方案状态
    if loaded_scheme_id:
        scheme_status_v = f"已加载方案: {loaded_scheme_id}（默认: {default_scheme_id or '-'}）"
    else:
        scheme_status_v = f"未加载模型（默认方案: {default_scheme_id or '未设置'}）"

    return scheme_select_update, scheme_desc_v, gpu_vram_v, rec_scheme_v, scheme_status_v


def _build_advanced_panel(engine_url: str, status: dict) -> tuple:
    """构建高级模式加速方案的 UI 数据。

    显示当前已加载方案的加速配置（只读展示，用户可修改后点击应用按钮）。

    Args:
        engine_url: 引擎 URL
        status: /status 返回的 dict

    Returns:
        (adv_sage, adv_lora, adv_fbc, adv_blockswap, adv_lora_scale, adv_status)
    """
    extra = status.get("extra", {})
    loaded_scheme_id = extra.get("loaded_scheme_id")

    if not loaded_scheme_id:
        # 未加载模型，显示默认值
        return False, False, False, False, 1.0, "未加载模型，加速方案未生效"

    # 从 /schemes 获取方案详情
    try:
        schemes_resp = _http_get(engine_url, "/schemes")
    except Exception as e:
        logger.error("获取方案详情失败: %s", e)
        return False, False, False, False, 1.0, f"获取方案详情失败: {e}"

    schemes = schemes_resp.get("schemes", [])
    loaded_scheme = next(
        (s for s in schemes if s.get("id") == loaded_scheme_id),
        None,
    )

    if not loaded_scheme:
        return False, False, False, False, 1.0, f"方案 {loaded_scheme_id} 详情不可用"

    # 由于 /schemes 接口当前不返回 acceleration 详情（只返回基本信息），
    # 这里用方案 ID 推断默认的加速配置
    # 方案 A: SageAttention + LightX2V 4-step + FirstBlockCache
    # 方案 B: SageAttention + LightX2V 8-step
    # 方案 C: BlockSwap + LightX2V 8-step + FirstBlockCache
    # 方案 D: 无加速方案
    sid = loaded_scheme_id.upper()
    if sid == "A":
        sage_v, lora_v, fbc_v, bs_v, scale_v = True, True, True, False, 1.0
        desc = "方案 A：SageAttention + LightX2V 4-step + FirstBlockCache"
    elif sid == "B":
        sage_v, lora_v, fbc_v, bs_v, scale_v = True, True, False, False, 1.0
        desc = "方案 B：SageAttention + LightX2V 8-step"
    elif sid == "C":
        sage_v, lora_v, fbc_v, bs_v, scale_v = False, True, True, True, 1.0
        desc = "方案 C：BlockSwap + LightX2V 8-step + FirstBlockCache"
    elif sid == "D":
        sage_v, lora_v, fbc_v, bs_v, scale_v = False, False, False, False, 1.0
        desc = "方案 D：无加速方案（BF16 原精度）"
    else:
        sage_v, lora_v, fbc_v, bs_v, scale_v = False, False, False, False, 1.0
        desc = f"未知方案 {sid}：使用默认值"

    return sage_v, lora_v, fbc_v, bs_v, scale_v, f"当前方案 {sid} 的加速配置：{desc}"


def update_scheme_description(engine_url: str, scheme_label: str) -> str:
    """v0.6+：根据下拉框选中项更新方案描述。

    Args:
        engine_url: 引擎 URL
        scheme_label: 下拉框选中项，格式 "A - 24G 显存流畅版（sdnq_uint4）"

    Returns:
        方案描述字符串
    """
    if not scheme_label:
        return "-"

    # 从 label 解析方案 ID
    sid = scheme_label.split(" - ")[0].strip() if " - " in scheme_label else ""

    if not sid:
        return "-"

    try:
        schemes_resp = _http_get(engine_url, "/schemes")
        schemes = schemes_resp.get("schemes", [])
        scheme = next((s for s in schemes if s.get("id") == sid), None)
        if not scheme:
            return f"方案 {sid} 不存在"

        return (
            f"{scheme.get('name', '')}\n"
            f"描述：{scheme.get('description', '')}\n"
            f"精度：{scheme.get('precision', '')} | "
            f"显存需求：{scheme.get('vram_required', '')} | "
            f"offload：{scheme.get('offload_level', '')}\n"
            f"模型仓库：{scheme.get('model_repo', '')}"
        )
    except Exception as e:
        return f"获取方案详情失败: {e}"


# v0.6.1+ 方案→参数映射（方案变化时联动 offload / steps / 加速方案）
# 与 schemes/*.yaml 实际配置对齐（v1.1：补齐 T4/T8，修正 A=30 步无 LoRA、C=L3、D=30 步）
_SCHEME_LINKAGE = {
    "T4": {"offload": "L2", "steps": 4,  "sage": False, "lora": True,  "fbc": False, "blockswap": False},
    "T8": {"offload": "L2", "steps": 8,  "sage": False, "lora": True,  "fbc": False, "blockswap": False},
    "A":  {"offload": "L2", "steps": 30, "sage": True,  "lora": False, "fbc": True,  "blockswap": False},
    "B":  {"offload": "L2", "steps": 8,  "sage": True,  "lora": True,  "fbc": False, "blockswap": False},
    "C":  {"offload": "L3", "steps": 8,  "sage": True,  "lora": True,  "fbc": True,  "blockswap": False},
    "D":  {"offload": "L0", "steps": 30, "sage": False, "lora": False, "fbc": False, "blockswap": False},
}


# ===== 功能 Tab → 范式映射（v0.6.2+ 双层交叉过滤用） =====
# T2V/I2V/FL2V 复用 FL2VA 范式模型；Ref2V/视频编辑 需要 Ref2VA 范式模型
_TAB_TO_PARADIGM = {
    MODE_T2V: "FL2VA",
    MODE_I2V: "FL2VA",
    MODE_FL2V: "FL2VA",
    MODE_REF2V: "Ref2VA",
    MODE_VIDEO_EDIT: "Ref2VA",
}


def cross_filter_variants(
    paradigm: Optional[str],
    scheme_id: Optional[str],
    caps_data: dict,
    schemes_data: dict,
) -> list:
    """v0.6.2+：双层交叉过滤模型变体（Tab→范式 × 方案→精度）。

    统一过滤函数，供 ``on_scheme_change_linkage`` 和 ``on_tab_change_filter``
    共同调用，保证两条事件链的过滤行为完全一致。

    过滤维度：
        1. **范式过滤**：根据当前功能 Tab 决定可用的 base_model
           - T2V / I2V / FL2V → FL2VA
           - Ref2V / 视频编辑 → Ref2VA
        2. **方案精度过滤**：根据当前方案 ID 的 ``applicable_variants``
           白名单过滤（如方案 A 仅适用 SDNQ 量化版）

    Args:
        paradigm: 范式过滤值（``"FL2VA"`` / ``"Ref2VA"``），``None`` 表示不按范式过滤
        scheme_id: 方案 ID（如 ``"A"``），``None`` / 空字符串表示不按方案过滤
        caps_data: ``/capabilities`` 返回数据（含 ``variants`` 列表）
        schemes_data: ``/schemes`` 返回数据（含 ``schemes`` + ``applicable_variants``）

    Returns:
        过滤后的 variant label 列表，每项格式：
        ``"[FL2VA] MiniMax/H3-FL2VA-SDNQ (768p, sdnq_uint4, 含音频)"``
    """
    # 1. 方案 → applicable_variants（精度过滤白名单）
    applicable_variants = set()
    if scheme_id:
        schemes_list = (schemes_data or {}).get("schemes", [])
        scheme_info = next((s for s in schemes_list if s.get("id") == scheme_id), None)
        if scheme_info:
            applicable_variants = set(scheme_info.get("applicable_variants", []))

    # 2. 双层交叉过滤 caps.variants
    caps_variants = (caps_data or {}).get("variants", [])
    labels: list = []
    for v in caps_variants:
        if not v.get("available", False):
            continue
        v_name = v.get("name", "")
        base_model = v.get("base_model", "")
        # 维度 1：范式过滤
        if paradigm and base_model != paradigm:
            continue
        # 维度 2：方案精度过滤
        if applicable_variants and v_name not in applicable_variants:
            continue
        precision = v.get("precision", "")
        resolution = v.get("resolution") or "未知"
        supports_audio = v.get("supports_audio", False)
        audio_tag = "含音频" if supports_audio else "无音频"
        func_tag = f"[{base_model}]" if base_model else ""
        label = f"{func_tag} {v_name} ({resolution}, {precision}, {audio_tag})".strip()
        labels.append(label)
    return labels


def on_scheme_change_linkage(
    engine_url: str, scheme_label: str, tab_id: str, caps_data: dict, schemes_data: dict
) -> tuple:
    """v0.6.2+：方案变化时联动更新模型变体 / offload / steps / 加速方案。

    联动逻辑：
        1. 从 scheme_label 解析方案 ID（如 "A - 24G 显存流畅版（sdnq_uint4）" → "A"）
        2. 从 schemes_data 获取该方案的 applicable_variants（精度过滤白名单）
        3. 调用 ``cross_filter_variants`` 双层交叉过滤：
           - 维度 1：当前功能 Tab → 范式（FL2VA / Ref2VA）
           - 维度 2：当前方案 ID → applicable_variants 精度白名单
        4. 根据方案 ID 设置 offload/steps/加速方案默认值

    Args:
        engine_url: 引擎 URL（用于获取方案描述，若 schemes_data 为空时回退到 API）
        scheme_label: 下拉框选中项，格式 "A - 24G 显存流畅版（sdnq_uint4）"
        tab_id: 当前功能 Tab ID（MODE_T2V / MODE_REF2V / ...），用于范式过滤
        caps_data: /capabilities 返回的数据（含 variants 列表）
        schemes_data: /schemes 返回的数据（含 schemes 列表 + applicable_variants）

    Returns:
        10 个字段：
            scheme_desc, variant_select_update, offload_select_update, steps_update,
            adv_sage, adv_lora, adv_fbc, adv_blockswap, adv_lora_scale, adv_status
    """
    if not scheme_label:
        return ("-", gr.update(), gr.update(), gr.update(),
                False, False, False, False, 1.0, "-")

    # 1. 解析方案 ID
    sid = scheme_label.split(" - ")[0].strip() if " - " in scheme_label else ""
    if not sid:
        return ("-", gr.update(), gr.update(), gr.update(),
                False, False, False, False, 1.0, "-")

    # 2. 获取方案详情 + applicable_variants
    schemes_list = (schemes_data or {}).get("schemes", [])
    scheme_info = next((s for s in schemes_list if s.get("id") == sid), None)

    # 若 schemes_data 为空，回退到 API 获取
    if not scheme_info:
        try:
            schemes_resp = _http_get(engine_url, "/schemes")
            schemes_list = schemes_resp.get("schemes", [])
            scheme_info = next((s for s in schemes_list if s.get("id") == sid), None)
        except Exception as e:
            logger.warning("获取方案详情失败: %s", e)

    # 3. 构造方案描述
    if scheme_info:
        scheme_desc = (
            f"{scheme_info.get('name', '')}\n"
            f"描述：{scheme_info.get('description', '')}\n"
            f"精度：{scheme_info.get('precision', '')} | "
            f"显存需求：{scheme_info.get('vram_required', '')} | "
            f"offload：{scheme_info.get('offload_level', '')}\n"
            f"模型仓库：{scheme_info.get('model_repo', '')}"
        )
    else:
        scheme_desc = f"方案 {sid} 详情不可用"

    # 4. 双层交叉过滤 variant（Tab→范式 × 方案→精度），统一调用 cross_filter_variants
    paradigm = _TAB_TO_PARADIGM.get(tab_id)
    variant_labels = cross_filter_variants(paradigm, sid, caps_data, schemes_data)

    if not variant_labels:
        variant_labels = ["该方案暂无可用模型，请到引擎详情页下载"]
        variant_default = None
    else:
        variant_default = variant_labels[0]

    variant_update = gr.update(choices=variant_labels, value=variant_default)

    # 5. 根据方案 ID 联动 offload / steps / 加速方案
    linkage = _SCHEME_LINKAGE.get(sid.upper(), {})
    offload_val = linkage.get("offload", "auto")
    steps_val = linkage.get("steps", 50)
    sage_v = linkage.get("sage", False)
    lora_v = linkage.get("lora", False)
    fbc_v = linkage.get("fbc", False)
    blockswap_v = linkage.get("blockswap", False)

    offload_update = gr.update(value=offload_val)
    steps_update = gr.update(value=steps_val)

    accel_desc = (
        f"方案 {sid} 联动：offload={offload_val}, steps={steps_val}, "
        f"Sage={sage_v}, LoRA={lora_v}, FBC={fbc_v}, BlockSwap={blockswap_v}"
    )

    return (
        scheme_desc, variant_update, offload_update, steps_update,
        sage_v, lora_v, fbc_v, blockswap_v, 1.0, accel_desc,
    )


def on_tab_change_filter(
    engine_url: str,
    tab_id: str,
    scheme_label: str,
    caps_data: dict,
    schemes_data: dict,
) -> tuple:
    """v0.6.2+：功能 Tab 切换时联动过滤模型变体 + 模型不匹配 toast 提示。

    联动逻辑：
        1. Tab → 范式映射（T2V/I2V/FL2V → FL2VA，Ref2V/视频编辑 → Ref2VA）
        2. 解析当前方案 ID（与 Tab 一起组成双层交叉过滤）
        3. 调用 ``cross_filter_variants`` 过滤 variant 列表
        4. 检查已加载模型是否匹配新 Tab 的范式：
           - 不匹配时调用 ``gr.Warning`` 显示 toast（不自动卸载，由用户决定）
           - 匹配时不提示
           - /status 不可达时静默跳过（不影响 Tab 切换）

    Args:
        engine_url: 引擎 URL（用于调用 /status 检查已加载模型）
        tab_id: 当前 Tab ID（MODE_T2V / MODE_I2V / MODE_FL2V / MODE_REF2V / MODE_VIDEO_EDIT）
        scheme_label: 当前选中的方案 label（如 "A - 24G 显存流畅版（sdnq_uint4）"）
        caps_data: /capabilities 返回数据
        schemes_data: /schemes 返回数据

    Returns:
        (variant_select_update, current_tab_value)
        - variant_select_update：``gr.update(choices=..., value=...)``
        - current_tab_value：传回 ``current_tab`` State 的 tab_id
    """
    # 1. Tab → 范式映射
    paradigm = _TAB_TO_PARADIGM.get(tab_id)

    # 2. 解析方案 ID
    sid = ""
    if scheme_label and " - " in scheme_label:
        sid = scheme_label.split(" - ")[0].strip()

    # 3. 双层交叉过滤（Tab→范式 × 方案→精度）
    labels = cross_filter_variants(paradigm, sid or None, caps_data, schemes_data)

    if not labels:
        labels = ["该功能暂无可用模型，请到引擎详情页下载"]
        default_label = None
    else:
        default_label = labels[0]

    variant_update = gr.update(choices=labels, value=default_label)

    # 4. 模型不匹配检查 + toast 提示（T4.3）
    #    /status 返回的 loaded_variant 形如 "MiniMax/H3-FL2VA-SDNQ"
    #    通过名称包含的范式关键字判断已加载模型所属范式
    try:
        status = _http_get(engine_url, "/status")
        loaded = (status or {}).get("loaded_variant", "")
        if loaded and paradigm:
            if "Ref2VA" in loaded:
                loaded_paradigm = "Ref2VA"
            elif "FL2VA" in loaded:
                loaded_paradigm = "FL2VA"
            else:
                loaded_paradigm = ""
            if loaded_paradigm and loaded_paradigm != paradigm:
                gr.Warning(
                    f"当前已加载模型 {loaded}（{loaded_paradigm}）不适用于 "
                    f"{tab_id} 功能，需加载 {paradigm} 范式模型。"
                    f"请在「模型变体」下拉框选择对应的 [{paradigm}] 模型后点击「加载模型」。"
                )
    except Exception as e:
        # /status 不可达时静默跳过，不阻塞 Tab 切换
        logger.debug("Tab 切换时检查 /status 失败（引擎可能未启动）: %s", e)

    return variant_update, tab_id


def save_default_scheme(engine_url: str, scheme_label: str) -> tuple:
    """v0.6+：保存默认方案（调用 POST /schemes/default）。

    Args:
        engine_url: 引擎 URL
        scheme_label: 下拉框选中项，格式 "A - 24G 显存流畅版（sdnq_uint4）"

    Returns:
        (scheme_status, log_message)
    """
    if not scheme_label:
        return "请先选择方案", "未选择方案"

    # 从 label 解析方案 ID
    sid = scheme_label.split(" - ")[0].strip() if " - " in scheme_label else scheme_label

    if not sid or sid not in ("A", "B", "C", "D"):
        return f"无效的方案 ID: {sid}", f"保存失败：无效的方案 ID {sid}"

    try:
        response = _http_post(
            engine_url, "/schemes/default",
            {"scheme_id": sid}, timeout=10.0,
        )
        msg = response.get("message", f"默认方案已设置为 {sid}")
        return f"✓ {msg}", f"默认方案已保存: {sid}"
    except Exception as e:
        return f"保存失败: {e}", f"保存默认方案失败: {e}"


def apply_acceleration_override(
    engine_url: str,
    variant_label: str,
    offload_level: str,
    adv_sage: bool,
    adv_lora: bool,
    adv_fbc: bool,
    adv_blockswap: bool,
    adv_lora_scale: float,
):
    """v0.6+：应用高级模式的加速方案覆盖（重新加载模型）。

    流程：
        1. 卸载当前模型（如已加载）
        2. 加载模型（携带 acceleration_override 参数）

    注意：当前后端 /load 不支持 acceleration_override 参数，
    此函数会通过 inputs.acceleration_override 字段传递，
    后端 engine.py._load_in_background 需读取并应用。

    由于当前后端尚未实现 acceleration_override 的解析，
    此函数暂只触发 unload + load，加速方案按当前默认方案生效。
    用户修改的加速方案会在后端支持后生效。

    Yields:
        (adv_status, status_display, variant_display, vram_display, offload_display, log_display)
    """
    variant = _parse_variant_label(variant_label)
    if not variant:
        yield (
            "请先选择 variant",
            "未选择 variant", "-", "-", "-",
            "未选择 variant，无法重新加载",
        )
        return

    # 构造加速方案覆盖描述
    override_desc_parts = []
    if adv_sage:
        override_desc_parts.append("SageAttention")
    if adv_lora:
        override_desc_parts.append(f"LightX2V LoRA(scale={adv_lora_scale})")
    if adv_fbc:
        override_desc_parts.append("FirstBlockCache")
    if adv_blockswap:
        override_desc_parts.append("BlockSwap")
    override_desc = ", ".join(override_desc_parts) if override_desc_parts else "无加速方案"

    yield (
        f"正在应用加速方案覆盖（{override_desc}）并重新加载模型...",
        "正在卸载...", "-", "-", "-",
        f"开始应用加速方案覆盖：{override_desc}",
    )

    # 1. 卸载当前模型
    try:
        _http_post(engine_url, "/unload", {}, timeout=60.0)
        yield (
            f"卸载完成，正在用 {override_desc} 重新加载...",
            "正在加载...", "-", "-", "-",
            f"卸载完成，开始加载 variant={variant}",
        )
    except Exception as e:
        yield (
            f"卸载失败: {e}",
            f"卸载失败: {e}", "-", "-", "-",
            f"卸载失败: {e}",
        )
        return

    # 2. 加载模型（携带 acceleration_override）
    # 注意：当前后端 /load 的 LoadRequest.inputs 字段是 dict，
    # 我们通过 inputs.acceleration_override 传递覆盖配置
    # 后端 engine.py._load_in_background 需读取此字段并应用
    try:
        body = {
            "variant": variant,
            "inputs": {
                "cpu_offload_level": offload_level,
                "acceleration_override": {
                    "sage_attention": {"enabled": adv_sage},
                    "lora": {
                        "enabled": adv_lora,
                        "scale": float(adv_lora_scale),
                    } if adv_lora else None,
                    "first_block_cache": {"enabled": adv_fbc},
                    "block_swap": {"enabled": adv_blockswap},
                },
            },
        }
        response = _http_post(engine_url, "/load", body, timeout=600.0)
        state = response.get("state", "?")
        loaded_variant = response.get("variant", variant)
        vram = response.get("vram_used", "-")
        if isinstance(vram, (int, float)):
            vram = f"{vram:.0f}MB"
        extra = response.get("extra", {})
        actual_offload = extra.get("offload_level", offload_level)

        yield (
            f"✓ 加速方案覆盖已应用（{override_desc}）",
            f"state={state} | 已加载",
            loaded_variant, vram, actual_offload,
            f"加载完成: variant={loaded_variant}, offload={actual_offload}, 加速方案: {override_desc}",
        )
    except Exception as e:
        yield (
            f"加载失败: {e}",
            f"加载失败: {e}", "-", "-", "-",
            f"应用加速方案覆盖失败: {e}",
        )


def load_variant(engine_url: str, variant_label: str, offload_level: str) -> tuple:
    """加载 variant。"""
    if not variant_label:
        return "请先选择 variant", "-", "-", "-", "未选择 variant"

    variant = _parse_variant_label(variant_label)

    try:
        body = {
            "variant": variant,
            "inputs": {
                "cpu_offload_level": offload_level,
            },
        }
        response = _http_post(engine_url, "/load", body, timeout=600.0)
        status_text = f"state={response.get('state', '?')}"
        variant_display = response.get("variant", variant)
        vram = response.get("vram_used", "-")
        if isinstance(vram, (int, float)):
            vram = f"{vram:.0f}MB"
        load_time = response.get("load_time", 0)
        # 从 extra 获取实际解析后的 offload 等级
        extra = response.get("extra", {})
        actual_offload = extra.get("offload_level", offload_level)
        log = (
            f"加载 {variant} 完成，耗时 {load_time:.2f}s，显存 {vram}，"
            f"offload 等级：{actual_offload}"
        )
        return status_text, variant_display, vram, actual_offload, log
    except Exception as e:
        logger.error("加载失败: %s", e)
        return f"加载失败: {e}", "-", "-", "-", f"加载失败: {e}"


def unload_variant(engine_url: str) -> tuple:
    """卸载模型。"""
    try:
        response = _http_post(engine_url, "/unload", {}, timeout=60.0)
        vram_freed = response.get("vram_freed", "0MB")
        return "state=idle", "-", "-", "-", f"卸载完成，释放显存 {vram_freed}"
    except Exception as e:
        return f"卸载失败: {e}", "-", "-", "-", f"卸载失败: {e}"


def lora_load(engine_url: str, name: str, weight: float) -> tuple:
    """加载 LoRA。"""
    if not name.strip():
        return "请输入 LoRA 名称", "LoRA 名称为空"
    try:
        body = {"name": name.strip(), "weight": float(weight)}
        response = _http_post(engine_url, "/lora/load", body, timeout=120.0)
        loaded = response.get("loaded_loras", [])
        status = f"已加载 {len(loaded)} 个 LoRA：{', '.join(loaded)}" if loaded else "LoRA 已加载"
        return status, f"LoRA {name} 加载成功（权重 {weight}）"
    except Exception as e:
        return f"加载失败: {e}", f"LoRA 加载失败: {e}"


def lora_unload(engine_url: str, name: str) -> tuple:
    """卸载 LoRA。"""
    if not name.strip():
        return "请输入要卸载的 LoRA 名称", "LoRA 名称为空"
    try:
        body = {"name": name.strip()}
        response = _http_post(engine_url, "/lora/unload", body, timeout=60.0)
        loaded = response.get("loaded_loras", [])
        status = f"已加载 {len(loaded)} 个 LoRA：{', '.join(loaded)}" if loaded else "无 LoRA 加载"
        return status, f"LoRA {name} 已卸载"
    except Exception as e:
        return f"卸载失败: {e}", f"LoRA 卸载失败: {e}"


def generate_video(
    engine_url: str,
    tab: str,
    variant_label: str,
    offload_level: str,
    t2v_prompt: str, t2v_negative: str,
    i2v_image: Optional[str], i2v_role: str, i2v_prompt: str, i2v_negative: str,
    fl2v_first: Optional[str], fl2v_last: Optional[str], fl2v_prompt: str, fl2v_negative: str,
    ref2v_images: list, ref2v_prompt: str, ref2v_negative: str,
    edit_video: Optional[str], edit_prompt: str, edit_negative: str,
    duration: int, fps: int, resolution: str, steps: int, guidance: float, seed: int,
):
    """流式视频生成。

    自动加载逻辑：
        1. 发送前检查 /status
        2. 如果未加载 → 自动 /load（带 offload_level）
        3. 加载完成后执行 /infer（SSE 流式）
    """
    log_lines: list[str] = []

    # ----- 1. 解析 variant 和模式参数 -----
    variant = _parse_variant_label(variant_label)
    if not variant:
        yield "请先选择模型", None, None, "-", "-", "-", "未选择 variant"
        return

    # 解析分辨率
    try:
        width, height = _parse_resolution(resolution)
    except Exception:
        width, height = 768, 432

    # ----- 2. 根据模式构造请求 -----
    images = []
    videos = []
    prompt = ""
    negative_prompt = ""

    if tab == MODE_T2V:
        prompt = t2v_prompt or ""
        negative_prompt = t2v_negative or ""
        if not prompt.strip():
            yield "请输入提示词", None, None, "-", "-", "-", "提示词为空"
            return
        log_lines.append(f"T2V 模式：prompt={prompt[:50]!r}")

    elif tab == MODE_I2V:
        prompt = i2v_prompt or ""
        negative_prompt = i2v_negative or ""
        if not i2v_image:
            yield "请上传图片", None, None, "-", "-", "-", "图片为空"
            return
        if not prompt.strip():
            yield "请输入提示词", None, None, "-", "-", "-", "提示词为空"
            return
        # 解析角色
        role = "first_frame" if i2v_role.startswith("first_frame") else "last_frame"
        images = [{"path": i2v_image, "role": role}]
        log_lines.append(f"I2V 模式：image={i2v_image}, role={role}, prompt={prompt[:50]!r}")

    elif tab == MODE_FL2V:
        prompt = fl2v_prompt or ""
        negative_prompt = fl2v_negative or ""
        if not fl2v_first or not fl2v_last:
            yield "请上传首帧和尾帧图片", None, None, "-", "-", "-", "首帧或尾帧为空"
            return
        if not prompt.strip():
            yield "请输入提示词", None, None, "-", "-", "-", "提示词为空"
            return
        images = [
            {"path": fl2v_first, "role": "first_frame"},
            {"path": fl2v_last, "role": "last_frame"},
        ]
        log_lines.append(f"FL2V 模式：first={fl2v_first}, last={fl2v_last}")

    elif tab == MODE_REF2V:
        prompt = ref2v_prompt or ""
        negative_prompt = ref2v_negative or ""
        if not ref2v_images:
            yield "请上传至少 1 张参考图", None, None, "-", "-", "-", "参考图为空"
            return
        if not prompt.strip():
            yield "请输入提示词", None, None, "-", "-", "-", "提示词为空"
            return
        # ref2v_images 是 gr.Gallery 返回的元组列表 [(path, caption), ...] 或路径列表
        paths = _extract_gallery_paths(ref2v_images)
        if len(paths) > 9:
            log_lines.append(f"警告：参考图超过 9 张，仅取前 9 张")
            paths = paths[:9]
        images = [{"path": p, "role": f"reference_{i+1}"} for i, p in enumerate(paths)]
        log_lines.append(f"Ref2V 模式：{len(paths)} 张参考图")

    elif tab == MODE_VIDEO_EDIT:
        prompt = edit_prompt or ""
        negative_prompt = edit_negative or ""
        if not edit_video:
            yield "请上传源视频", None, None, "-", "-", "-", "源视频为空"
            return
        if not prompt.strip():
            yield "请输入编辑指令", None, None, "-", "-", "-", "编辑指令为空"
            return
        videos = [{"path": edit_video, "role": "source_video"}]
        log_lines.append(f"VideoEdit 模式：source={edit_video}, prompt={prompt[:50]!r}")

    else:
        yield f"未知模式: {tab}", None, None, "-", "-", "-", f"未知模式: {tab}"
        return

    # ----- 3. 模型加载状态校验（v0.4.x：先判断引擎状态，决定是否拦截） -----
    try:
        status = _http_get(engine_url, "/status")
    except Exception as e:
        log_lines.append(f"获取状态失败: {e}")
        yield "获取状态失败", None, None, "-", "-", "-", "\n".join(log_lines)
        raise gr.Error(
            f"无法连接引擎获取状态：{e}\n\n"
            f"请检查引擎是否正常运行（URL: {engine_url}）。",
            duration=10,
        )

    current_state = status.get("state", "unknown")
    # v0.4.x：直接使用 status 字段，不再做猜测性路径处理
    loaded_variant = status.get("loaded_variant") or ""

    # v0.4.x：引擎处于 ERROR 状态时直接拦截，弹窗提示，阻止生成
    if current_state == "error":
        err_msg = "引擎内部错误（请到引擎详情页查看错误详情）"
        log_lines.append(f"引擎处于错误状态: {err_msg}")
        yield "引擎错误", None, None, "-", "-", "-", "\n".join(log_lines)
        raise gr.Error(
            f"引擎处于错误状态：{err_msg}\n\n"
            f"请到引擎详情页查看错误详情，或重启引擎后重试。",
            duration=10,
        )

    # ----- 4. 自动加载 / 切换 -----
    if current_state != "ready" or not loaded_variant:
        log_lines.append(f"模型未加载（state={current_state}），自动加载 {variant}（offload={offload_level}）")
        yield "正在加载模型...", None, None, "-", "-", "-", "\n".join(log_lines)
        try:
            load_resp = _http_post(
                engine_url, "/load",
                {"variant": variant, "inputs": {"cpu_offload_level": offload_level}},
                timeout=600.0,
            )
            # /load 是 fire-and-forget，立即返回 state="loading"
            # 必须轮询 /status 直到 ready 才能发 /infer，否则触发 loading → inferring
            yield from _wait_for_load_complete(engine_url, load_resp, log_lines)
            yield "模型加载完成，开始推理...", None, None, "-", "-", "-", "\n".join(log_lines)
        except Exception as e:
            log_lines.append(f"自动加载失败: {e}")
            yield "加载失败", None, None, "-", "-", "-", "\n".join(log_lines)
            raise gr.Error(
                f"模型自动加载失败：{e}\n\n"
                f"变体: {variant}\n"
                f"offload 等级: {offload_level}\n\n"
                f"可能原因：\n"
                f"1. 模型文件不完整（请到引擎详情页检查文件状态）\n"
                f"2. GPU 显存不足（请降低 offload 等级或选择 int8 量化版）\n"
                f"3. 引擎内部错误（请查看引擎日志）",
                duration=15,
            )
    elif loaded_variant != variant:
        log_lines.append(f"切换模型: {loaded_variant} → {variant}")
        yield "正在切换模型...", None, None, "-", "-", "-", "\n".join(log_lines)
        try:
            _http_post(engine_url, "/unload", {}, timeout=60.0)
            load_resp = _http_post(
                engine_url, "/load",
                {"variant": variant, "inputs": {"cpu_offload_level": offload_level}},
                timeout=600.0,
            )
            # /load fire-and-forget，轮询 /status 直到 ready
            yield from _wait_for_load_complete(engine_url, load_resp, log_lines)
            log_lines.append("切换完成")
            yield "模型切换完成，开始推理...", None, None, "-", "-", "-", "\n".join(log_lines)
        except Exception as e:
            log_lines.append(f"切换失败: {e}")
            yield "切换失败", None, None, "-", "-", "-", "\n".join(log_lines)
            raise gr.Error(
                f"模型切换失败：{e}\n\n"
                f"已加载: {loaded_variant}\n"
                f"目标: {variant}\n"
                f"offload 等级: {offload_level}\n\n"
                f"请到引擎详情页查看错误详情。",
                duration=15,
            )
    else:
        log_lines.append(f"模型已加载（variant={loaded_variant}），直接推理")
        yield "模型已加载，开始推理...", None, None, "-", "-", "-", "\n".join(log_lines)

    # ----- 4. 构造推理请求 -----
    request_body = {
        "variant": variant,
        "inputs": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "task_type": tab,
            "duration": int(duration),
            "fps": int(fps),
            "width": int(width),
            "height": int(height),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "seed": int(seed) if seed is not None else -1,
            "cpu_offload_level": offload_level,
        },
        "images": images,
        "videos": videos,
    }

    # ----- 6. SSE 流式推理 -----
    try:
        with httpx.Client(timeout=3600.0) as client:  # 视频推理可能很慢
            with client.stream(
                "POST",
                f"{engine_url}/infer",
                json=request_body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        frame = json.loads(data_str)
                    except Exception:
                        continue

                    stage = frame.get("stage", "")
                    progress = frame.get("progress", 0.0)

                    # 视频生成阶段进度
                    if stage == "video.start":
                        msg = f"开始生成（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "-", "\n".join(log_lines)

                    elif stage == "video.model_ready":
                        msg = f"模型就绪（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "-", "\n".join(log_lines)

                    elif stage == "video.prompt_processing":
                        msg = f"处理提示词（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "-", "\n".join(log_lines)

                    elif stage == "video.generating":
                        step_n = frame.get("step", 0)
                        total_steps = frame.get("total_steps", 50)
                        msg = f"生成视频帧中... {step_n}/{total_steps}（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "-", "\n".join(log_lines)

                    elif stage == "video.vae_decoding":
                        msg = f"VAE 解码视频（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "-", "\n".join(log_lines)

                    elif stage == "video.audio_decoding":
                        msg = f"解码音频（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "-", "\n".join(log_lines)

                    elif stage == "video.saving":
                        msg = f"保存文件中（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "-", "\n".join(log_lines)

                    elif stage == "completed" or stage == "video.completed":
                        result = frame.get("result", {})
                        timing = frame.get("timing", {})
                        video_url = result.get("video_url", "")
                        audio_url = result.get("audio_url", "")
                        video_only_url = result.get("video_only_url", "")
                        duration_ret = result.get("duration", 0)
                        mode = result.get("mode", tab)
                        actual_offload = result.get("offload_level", offload_level)
                        is_placeholder = result.get("is_placeholder", False)

                        # 拼完整 URL（用于显示下载链接文本）
                        full_video_url = f"{engine_url}{video_url}" if video_url else ""
                        full_audio_url = f"{engine_url}{audio_url}" if audio_url else ""
                        full_video_only_url = f"{engine_url}{video_only_url}" if video_only_url else ""

                        # v0.4.x 修复：Gradio safehttpx 拦截 127.0.0.1（"Hostname 127.0.0.1 failed validation"）
                        # 优先使用本地文件绝对路径传给 gr.Video / gr.Audio，绕过 safehttpx
                        # 仅在本地文件不存在时才回退到 URL（极少数情况）
                        gradio_video_value = _url_to_local_path(video_url) or (full_video_url if full_video_url else None)
                        gradio_audio_value = _url_to_local_path(audio_url) or (full_audio_url if full_audio_url else None)

                        # 元信息
                        meta_parts = [
                            f"模式: {mode}",
                            f"时长: {duration_ret:.2f}s",
                            f"offload: {actual_offload}",
                            f"seed: {seed}",
                        ]
                        if is_placeholder:
                            meta_parts.append("⚠️ placeholder 实现（未真正生成视频）")
                        meta_str = " | ".join(meta_parts)

                        total_seconds = timing.get("total_seconds", 0)
                        log_lines.append(
                            f"完成: 模式={mode}, 时长 {duration_ret:.2f}s, "
                            f"总耗时 {total_seconds:.2f}s, offload={actual_offload}"
                        )
                        yield "完成", gradio_video_value, gradio_audio_value, full_video_url or "-", full_audio_url or "-", meta_str, "\n".join(log_lines)
                        return

                    elif stage == "error" or stage == "video.error":
                        err = frame.get("error", {})
                        msg = f"错误: {err.get('message', '?')}"
                        log_lines.append(msg)
                        yield msg, None, None, "-", "-", "-", "\n".join(log_lines)
                        return

    except Exception as e:
        logger.exception("MiniMax-H3 推理失败")
        yield f"推理异常: {e}", None, None, "-", "-", "-", f"推理异常: {e}"


# ===== 工具函数 =====

def _parse_resolution(resolution: str) -> tuple[int, int]:
    """解析分辨率字符串 '768x432 (16:9)' → (768, 432)。"""
    # 提取 "768x432" 部分
    size_part = resolution.split(" ")[0]
    w, h = size_part.split("x")
    return int(w), int(h)


def _extract_gallery_paths(gallery_value) -> list[str]:
    """从 gr.Gallery 返回值提取文件路径列表。

    gr.Gallery 返回值可能为：
        - [(path, caption), ...]    # Gradio 6.0+ 带标题
        - [path, ...]               # 仅路径
    """
    if not gallery_value:
        return []
    paths = []
    for item in gallery_value:
        if isinstance(item, (list, tuple)):
            paths.append(item[0])
        else:
            paths.append(item)
    return paths


def _http_get(base_url: str, path: str) -> dict:
    """GET 请求。"""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{base_url}{path}")
        response.raise_for_status()
        return response.json()


def _http_post(base_url: str, path: str, json_body: dict, timeout: float = 600.0) -> dict:
    """POST 请求。"""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base_url}{path}", json=json_body)
        response.raise_for_status()
        return response.json()


def _wait_for_load_complete(engine_url: str, load_response: dict, log_lines: list):
    """轮询 /status 直到模型加载完成（ready/error/超时）。

    /load 端点是 fire-and-forget 模式，立即返回 state="loading"。
    调用方拿到响应后必须用本 generator 轮询直到 ready 才能发 /infer，
    否则会触发 loading → inferring 非法状态转换。

    Yields 进度提示元组（与外层 generator 的 yield 元组结构一致）。
    在 ready 时 return；error/被中断/超时时 raise RuntimeError。
    """
    # 兼容同步加载模式（state != "loading" 表示后端已加载完成）
    if load_response.get("state") != "loading":
        log_lines.append("加载完成（同步模式）")
        return

    max_wait, poll_interval = 600, 2
    waited = 0
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        try:
            status = _http_get(engine_url, "/status")
        except Exception as e:
            log_lines.append(f"轮询状态失败（继续重试）: {e}")
            continue
        cur_state = status.get("state", "unknown")
        if cur_state == "ready":
            log_lines.append(f"模型加载完成（耗时 {waited}s）")
            return
        if cur_state == "error":
            err = status.get("error") or status.get("error_message") or "未知错误"
            raise RuntimeError(f"模型加载失败（{waited}s）：{err}")
        if cur_state in ("idle", "stopped"):
            raise RuntimeError(f"模型加载被中断（{waited}s，state={cur_state}）")
        # 仍在 loading，每 10s 推送进度
        if waited % 10 == 0:
            yield f"正在加载模型... ({waited}s)", None, None, "-", "-", "-", "\n".join(log_lines)
    raise RuntimeError(f"模型加载超时（{max_wait}s）")
