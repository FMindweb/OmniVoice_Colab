import os
import sys
import logging
import tempfile
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile
import re
import uuid

# ---------------------------------------------------------------------------
# 路径与环境配置
# ---------------------------------------------------------------------------
temp_audio_dir = "./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)
from subtitle import subtitle_maker

try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# 翻译字典
# ---------------------------------------------------------------------------
I18N = {
    "cn": {
        "title": "🎙️ OmniVoice 多语言语音合成",
        "subtitle": "支持 600 多种语言，支持声音克隆与人声设计。",
        "sub_btn": "📻 更多好内容在这里",
        "tab_clone": "声音克隆",
        "tab_design": "人声设计",
        "text_label": "待合成文本",
        "text_placeholder": "输入想要转为语音的文字...",
        "tags_label": "插入情感/事件标签：",
        "lang_label": "语言选择 (可选)",
        "auto": "自动检测",
        "subs_label": "生成字幕文件",
        "ref_audio_label": "参考音频 (3-10秒清晰人声)",
        "ref_text_label": "参考文本",
        "ref_text_placeholder": "自动识别或手动输入...",
        "gen_btn": "开始合成",
        "settings_label": "高级生成设置",
        "speed_label": "语速",
        "duration_label": "固定时长 (秒)",
        "steps_label": "推理步数",
        "denoise_label": "启用去噪",
        "cfg_label": "引导强度 (CFG)",
        "output_label": "合成结果",
        "status_label": "状态信息",
        "download_label": "下载成果物",
        "gender": "性别",
        "age": "年龄",
        "pitch": "音高",
        "style": "风格",
        "accent": "英语口音",
        "dialect": "中文方言"
    },
    "en": {
        "title": "🎙️ OmniVoice Multilingual TTS",
        "subtitle": "600+ languages supported, featuring Voice Clone & Design.",
        "sub_btn": "📻 More Funs Here",
        "tab_clone": "Voice Clone",
        "tab_design": "Voice Design",
        "text_label": "Text to Synthesize",
        "text_placeholder": "Enter text here...",
        "tags_label": "Insert Event Tags:",
        "lang_label": "Language (Optional)",
        "auto": "Auto",
        "subs_label": "Generate Subtitles",
        "ref_audio_label": "Reference Audio (3-10s)",
        "ref_text_label": "Reference Text",
        "ref_text_placeholder": "Auto-transcribed or manual entry...",
        "gen_btn": "Generate",
        "settings_label": "Advanced Settings",
        "speed_label": "Speed",
        "duration_label": "Duration (sec)",
        "steps_label": "Inference Steps",
        "denoise_label": "Denoise",
        "cfg_label": "Guidance Scale (CFG)",
        "output_label": "Output Audio",
        "status_label": "Status",
        "download_label": "Download Files",
        "gender": "Gender",
        "age": "Age",
        "pitch": "Pitch",
        "style": "Style",
        "accent": "English Accent",
        "dialect": "Chinese Dialect"
    }
}

# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
print("Loading model...")
# (此处省略你原本的载入逻辑，假设 model 已定义)
# model = ... 
# sampling_rate = model.sampling_rate

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _build_instruct(groups):
    # 简化的映射逻辑
    INSTRUCT_MAP = {"男": "Male", "女": "Female", "孩童": "Child", "青少年": "Teenager", "青年": "Young Adult"}
    selected = [INSTRUCT_MAP.get(g, g) for g in groups if g and g not in ["自动检测", "Auto"]]
    return ", ".join(selected) if selected else None

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
css = """
.gradio-container {max-width: 100% !important;}
.tag-container { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
.tag-btn { min-width: 60px !important; height: 30px !important; font-size: 12px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    lang_state = gr.State("cn")
    
    with gr.Row():
        gr.Markdown("") # 占位
        lang_switch = gr.Radio(choices=[("中文", "cn"), ("English", "en")], value="cn", label="Interface Language / 界面语言")
        gr.Markdown("") # 占位

    # 标题部分
    with gr.Column(elem_id="header"):
        title_md = gr.HTML(f"<div style='text-align:center'><h1>{I18N['cn']['title']}</h1><p>{I18N['cn']['subtitle']}</p></div>")
        sub_btn_html = gr.HTML(f"<div style='text-align:center; margin-bottom:20px;'><a href='#' style='background:linear-gradient(90deg, #4F46E5, #7C3AED); color:white; padding:10px 20px; border-radius:50px; text-decoration:none;'>{I18N['cn']['sub_btn']}</a></div>")

    with gr.Tabs() as main_tabs:
        # --- 克隆页面 ---
        with gr.TabItem(I18N['cn']['tab_clone'], id="clone_tab") as vc_tab:
            with gr.Row():
                with gr.Column():
                    vc_text = gr.Textbox(label=I18N['cn']['text_label'], lines=4, placeholder=I18N['cn']['text_placeholder'], elem_id="vc_textbox")
                    vc_lang = gr.Dropdown(label=I18N['cn']['lang_label'], choices=["自动检测"] + sorted(LANG_NAMES), value="自动检测")
                    vc_want_subs = gr.Checkbox(label=I18N['cn']['subs_label'], value=False)
                    vc_ref_audio = gr.Audio(label=I18N['cn']['ref_audio_label'], type="filepath")
                    vc_ref_text = gr.Textbox(label=I18N['cn']['ref_text_label'], placeholder=I18N['cn']['ref_text_placeholder'])
                    vc_btn = gr.Button(I18N['cn']['gen_btn'], variant="primary")
                with gr.Column():
                    vc_audio = gr.Audio(label=I18N['cn']['output_label'])
                    vc_status = gr.Textbox(label=I18N['cn']['status_label'])

        # --- 设计页面 ---
        with gr.TabItem(I18N['cn']['tab_design'], id="design_tab") as vd_tab:
            with gr.Row():
                with gr.Column():
                    vd_text = gr.Textbox(label=I18N['cn']['text_label'], lines=4, placeholder=I18N['cn']['text_placeholder'], elem_id="vd_textbox")
                    with gr.Row():
                        vd_lang = gr.Dropdown(label=I18N['cn']['lang_label'], choices=["自动检测"] + sorted(LANG_NAMES), value="自动检测")
                        vd_want_subs = gr.Checkbox(label=I18N['cn']['subs_label'], value=False)
                    vd_btn = gr.Button(I18N['cn']['gen_btn'], variant="primary")
                    with gr.Accordion(I18N['cn']['settings_label'], open=True) as vd_acc:
                        gender_drop = gr.Dropdown(label=I18N['cn']['gender'], choices=["自动检测", "男", "女"], value="女")
                        age_drop = gr.Dropdown(label=I18N['cn']['age'], choices=["自动检测", "青年", "中年", "老年"], value="青年")
                with gr.Column():
                    vd_audio = gr.Audio(label=I18N['cn']['output_label'])
                    vd_status = gr.Textbox(label=I18N['cn']['status_label'])

    # ---------------------------------------------------------------------------
    # 核心：语言切换逻辑
    # ---------------------------------------------------------------------------
    def change_language(lang):
        d = I18N[lang]
        # 返回一系列 update 对象来改变界面
        return [
            # 标题
            f"<div style='text-align:center'><h1>{d['title']}</h1><p>{d['subtitle']}</p></div>",
            f"<div style='text-align:center; margin-bottom:20px;'><a href='#' style='background:linear-gradient(90deg, #4F46E5, #7C3AED); color:white; padding:10px 20px; border-radius:50px; text-decoration:none;'>{d['sub_btn']}</a></div>",
            # Tab 标签
            gr.update(label=d['tab_clone']),
            gr.update(label=d['tab_design']),
            # 克隆页组件
            gr.update(label=d['text_label'], placeholder=d['text_placeholder']),
            gr.update(label=d['lang_label']),
            gr.update(label=d['subs_label']),
            gr.update(label=d['ref_audio_label']),
            gr.update(label=d['ref_text_label'], placeholder=d['ref_text_placeholder']),
            gr.update(value=d['gen_btn']),
            gr.update(label=d['output_label']),
            gr.update(label=d['status_label']),
            # 设计页组件
            gr.update(label=d['text_label'], placeholder=d['text_placeholder']),
            gr.update(label=d['lang_label']),
            gr.update(label=d['subs_label']),
            gr.update(value=d['gen_btn']),
            gr.update(label=d['output_label']),
            gr.update(label=d['status_label']),
            gr.update(label=d['gender']),
            gr.update(label=d['age'])
        ]

    lang_switch.change(
        change_language,
        inputs=[lang_switch],
        outputs=[
            title_md, sub_btn_html, vc_tab, vd_tab,
            vc_text, vc_lang, vc_want_subs, vc_ref_audio, vc_ref_text, vc_btn, vc_audio, vc_status,
            vd_text, vd_lang, vd_want_subs, vd_btn, vd_audio, vd_status, gender_drop, age_drop
        ]
    )

if __name__ == "__main__":
    demo.launch()