# # %cd /content/omnivoice-colab
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
import os
import uuid

temp_audio_dir="./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 设置路径以从 /content/omnivoice-colab/OmniVoice/ 导入 subtitle_maker
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)
from subtitle import subtitle_maker

# 尝试导入 Whisper 支持的语言字典
try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# 日志设置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# 模型加载 (全局作用域)
# ---------------------------------------------------------------------------
print("正在从 k2-fsa/OmniVoice 加载模型至 CUDA ...")

from hf_mirror import download_model

try:
  model = OmniVoice.from_pretrained(
      "k2-fsa/OmniVoice",
      device_map="cuda",
      dtype=torch.float16,
      load_asr=False,
  )
except Exception as e:
  omnivoice_model_path=download_model(
    "k2-fsa/OmniVoice",
    download_folder="./OmniVoice_Model",
    redownload=False,
    workers=6,
    use_snapshot=False,
  )

  model = OmniVoice.from_pretrained(
      omnivoice_model_path,
      device_map="cuda",
      dtype=torch.float16,
      load_asr=False,
  )
sampling_rate = model.sampling_rate
print("模型加载成功！")

# ---------------------------------------------------------------------------
# 情感/事件标签 & JS 函数
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[笑声]", "[叹气]", "[确认-英]", "[疑问-英]", 
    "[疑问-啊]", "[疑问-哦]", "[疑问-诶]", "[疑问-咦]",
    "[惊讶-啊]", "[惊讶-哦]", "[惊讶-哇]", "[惊讶-哟]", 
    "[不满-唔]"
]

# 对应的实际模型标签映射 (如果模型只识别英文标签，请保留此映射)
TAG_MAP = {
    "[笑声]": "[laughter]", "[叹气]": "[sigh]", "[确认-英]": "[confirmation-en]", 
    "[疑问-英]": "[question-en]", "[疑问-啊]": "[question-ah]", "[疑问-哦]": "[question-oh]", 
    "[疑问-诶]": "[question-ei]", "[疑问-咦]": "[question-yi]", "[惊讶-啊]": "[surprise-ah]", 
    "[惊讶-哦]": "[surprise-oh]", "[惊讶-哇]": "[surprise-wa]", "[惊讶-哟]": "[surprise-yo]", 
    "[不满-唔]": "[dissatisfaction-hnn]"
}

# JS 用于在光标处插入标签
INSERT_TAG_JS = """
(tag_val, current_text) => {
    const el_id = tag_val.includes('vc') ? '#vc_textbox' : '#vd_textbox';
    const textarea = document.querySelector(el_id + ' textarea');
    
    // 这里将中文显示的标签转回模型识别的英文标签
    const tag_mapping = {
        "[笑声]": "[laughter]", "[叹气]": "[sigh]", "[确认-英]": "[confirmation-en]", 
        "[疑问-英]": "[question-en]", "[疑问-啊]": "[question-ah]", "[疑问-哦]": "[question-oh]", 
        "[疑问-诶]": "[question-ei]", "[疑问-咦]": "[question-yi]", "[惊讶-啊]": "[surprise-ah]", 
        "[惊讶-哦]": "[surprise-oh]", "[惊讶-哇]": "[surprise-wa]", "[惊讶-哟]": "[surprise-yo]", 
        "[不满-唔]": "[dissatisfaction-hnn]"
    };
    const real_tag = tag_mapping[tag_val] || tag_val;

    if (!textarea) return current_text + " " + real_tag;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return real_tag;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + real_tag + suffix + current_text.slice(end);
}
"""

# ---------------------------------------------------------------------------
# UI 配置 & 语言映射
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["自动检测"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_CATEGORIES = {
    "性别": ["男", "女"],
    "年龄": ["孩童", "青少年", "青年", "中年", "老年"],
    "音高": ["极低音", "低音", "中等音高", "高音", "极高音"],
    "风格": ["耳语/悄悄话"],
    "英语口音": [
        "美式口音", "英式口音", "澳洲口音", "中式英语",
        "加拿大口音", "印度口音", "韩式英语", "葡萄牙口音",
        "俄式口音", "日式英语"
    ],
    "中文方言": [
        "河南话", "陕西话", "四川话", "贵州话",
        "云南话", "桂林话", "济南话", "石家庄话",
        "甘肃话", "宁夏话", "青岛话", "东北话"
    ],
}

# 内部参数映射（将中文UI选项转为模型指令）
INSTRUCT_MAP = {
    "男": "Male", "女": "Female",
    "孩童": "Child", "青少年": "Teenager", "青年": "Young Adult", "中年": "Middle-aged", "老年": "Elderly",
    "极低音": "Very Low Pitch", "低音": "Low Pitch", "中等音高": "Moderate Pitch", "高音": "High Pitch", "极高音": "Very High Pitch",
    "耳语/悄悄话": "Whisper",
    "美式口音": "American Accent", "英式口音": "British Accent", "澳洲口音": "Australian Accent",
    "中式英语": "Chinese Accent", "加拿大口音": "Canadian Accent", "印度口音": "Indian Accent",
    "韩式英语": "Korean Accent", "葡萄牙口音": "Portuguese Accent", "俄式口音": "Russian Accent", "日式英语": "Japanese Accent"
}

_ATTR_INFO = {
    "英语口音": "仅对英语语音有效。",
    "中文方言": "仅对中文语音有效。",
}

# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------
def _is_whisper_supported(lang):
    if not lang or lang == "自动检测":
        return True 
    if WHISPER_LANGUAGE_CODE is None:
        return True 
    supported_langs = [str(k).lower() for k in WHISPER_LANGUAGE_CODE.keys()] + \
                      [str(v).lower() for v in WHISPER_LANGUAGE_CODE.values()]
    lang_lower = lang.lower()
    for w_lang in supported_langs:
        if w_lang in lang_lower or lang_lower in w_lang:
            return True
    return False

def generate_subtitles_if_needed(wav_path, lang, want_subs):
    if not want_subs:
        return None, None, None
    if not _is_whisper_supported(lang):
        logging.warning(f"语言 '{lang}' 可能不支持 Whisper。跳过字幕生成。")
        return None, None, None
    try:
        whisper_lang = lang if (lang and lang != "自动检测") else None
        whisper_results = subtitle_maker(wav_path, whisper_lang)
        if whisper_results and len(whisper_results) > 3:
            return whisper_results[1], whisper_results[2], whisper_results[3] 
    except Exception as e:
        logging.warning(f"字幕生成失败: {e}")
    return None, None, None

def tts_file_name(text, language="en"):
    global temp_audio_dir
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    clean_text = clean_text.lower().strip().replace(" ", "_")
    if not clean_text: clean_text = "audio"
    truncated = clean_text[:20]
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"
    rand = uuid.uuid4().hex[:8].upper()
    return f"{temp_audio_dir}/{truncated}_{lang}_{rand}.wav"

def _gen_core(
    text, language, ref_audio, instruct, num_step, guidance_scale, 
    denoise, speed, duration, preprocess_prompt, postprocess_output, mode, ref_text=None
):
    if not text or not text.strip():
        return None, "请输入要合成的文本。"

    if mode == "clone" and ref_audio and not ref_text:
        try:
            whisper_lang = language if (language and language != "自动检测") else None
            whisper_results = subtitle_maker(ref_audio, whisper_lang)
            if whisper_results and len(whisper_results) > 7:
                ref_text = whisper_results[7]
        except Exception as e:
            logging.warning(f"自动转录失败: {e}")

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "自动检测") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            return None, "请上传参考音频。"
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"错误: {type(e).__name__}: {e}"

    waveform = audio[0].squeeze(0).numpy()
    waveform = (waveform * 32767).astype(np.int16)
    return (sampling_rate, waveform), "生成完毕。"

# ---------------------------------------------------------------------------
# Gradio UI 构建
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(font=["Inter", "Microsoft YaHei", "sans-serif"])
css = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 2.2em !important; font-weight: 800 !important;}
.compact-audio audio {height: 60px !important;}
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
}
.tag-btn {
    min-width: fit-content !important;
    height: 30px !important;
    font-size: 12px !important;
    background: #f3f4f6 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 4px !important;
    padding: 0 8px !important;
}
"""

def _lang_dropdown(label="语言选择 (可选)", value="自动检测"):
    return gr.Dropdown(
        label=label, choices=_ALL_LANGUAGES, value=value,
        allow_custom_value=False, interactive=True,
    )

def _gen_settings():
    with gr.Accordion("高级生成设置 (可选)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="语速", info="1.0 为正常速度。>1 变快，<1 变慢。")
        du = gr.Number(value=None, label="固定时长 (秒)", info="设置后将忽略语速设置。")
        ns = gr.Slider(4, 64, value=32, step=1, label="推理步数", info="步数越低速度越快，越高音质越好。")
        dn = gr.Checkbox(label="启用去噪", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="引导强度 (CFG)")
        pp = gr.Checkbox(label="预处理提示词", value=True, info="自动去除静音并修整参考音频。")
        po = gr.Checkbox(label="后处理输出", value=True, info="自动去除生成音频末尾的冗长静音。")
    return ns, gs, dn, sp, du, pp, po

with gr.Blocks(theme=theme, css=css, title="OmniVoice 多语言演示") as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ OmniVoice 多语言语音合成</h1>
            <p style="font-size: 1.2em; color: #666; margin-bottom: 15px;">支持 600 多种语言的 SOTA 语音模型，提供 <b>声音克隆</b> 与 <b>人声设计</b> 功能。</p>
            
            <div style="margin-bottom: 20px;">
                <a href="https://www.youtube.com/@fm19.2?sub_confirmation=1" target="_blank" style="
                    background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
                    color: white;
                    padding: 10px 24px;
                    border-radius: 50px;
                    text-decoration: none;
                    font-weight: bold;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
                    transition: transform 0.2s ease;
                " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    <span>📻更多好内容在这里</span>
                </a>
            </div>
        </div>
    """)

    with gr.Tabs():
        # ==============================================================
        # 声音克隆
        # ==============================================================
        with gr.TabItem("声音克隆 (Voice Clone)"):
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(label="待合成文本", lines=4, placeholder="输入想要转为语音的文字...", elem_id="vc_textbox")
                    
                    gr.Markdown("**插入情感/事件标签：**")
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(fn=None, inputs=[btn, vc_text], outputs=vc_text, js=INSERT_TAG_JS)

                    with gr.Row():
                      vc_lang = _lang_dropdown()
                      vc_want_subs = gr.Checkbox(label="生成字幕文件", value=False)
                    
                    vc_ref_audio = gr.Audio(label="参考音频 (上传 3-10 秒清晰人声)", type="filepath", elem_classes="compact-audio")
                    vc_ref_text = gr.Textbox(
                        label="参考音频文本内容", lines=2, 
                        placeholder="上传音频后将自动识别。如果识别有误，请手动修改以提升克隆效果。"
                    )
                                        
                    vc_btn = gr.Button("开始合成", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="合成结果", type="numpy")
                    vc_status = gr.Textbox(label="状态信息", lines=1)
                    
                    with gr.Accordion("下载内容", open=False):
                        vc_out_wav = gr.File(label="音频文件 (WAV)")
                        vc_out_custom_srt = gr.File(label="句子级字幕 (SRT)")
                        vc_out_word_srt = gr.File(label="词级字幕 (SRT)")
                        vc_out_shorts_srt = gr.File(label="短视频格式字幕 (SRT)")

            vc_ref_audio.change(
                fn=lambda aud, lng: gr.update(value=subtitle_maker(aud, lng if lng!="自动检测" else None)[7] if aud else ""),
                inputs=[vc_ref_audio, vc_lang],
                outputs=[vc_ref_text]
            )

            def _clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po):
                res = _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text)
                if res[0] is None: return None, res[1], None, None, None, None
                
                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vc_btn.click(
                _clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                outputs=[vc_audio, vc_status, vc_out_wav, vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt],
            )

        # ==============================================================
        # 人声设计
        # ==============================================================
        with gr.TabItem("人声设计 (Voice Design)"):
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(label="待合成文本", lines=4, placeholder="输入想要转为语音的文字...", elem_id="vd_textbox")
                    
                    gr.Markdown("**插入情感/事件标签：**")
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(fn=None, inputs=[btn, vd_text], outputs=vd_text, js=INSERT_TAG_JS)

                    with gr.Row():
                      vd_lang = _lang_dropdown(value='自动检测')
                      vd_want_subs = gr.Checkbox(label="生成字幕文件", value=False)
                    
                    vd_btn = gr.Button("开始合成", variant="primary")
                    
                    with gr.Accordion("角色声音定制", open=True):
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            default_val = "自动检测"
                            if _cat == "性别": default_val = "女"
                            elif _cat == "年龄": default_val = "青年"
                                
                            vd_groups.append(
                                gr.Dropdown(label=_cat, choices=["自动检测"] + _choices, value=default_val, info=_ATTR_INFO.get(_cat))
                            )
                        
                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="合成结果", type="numpy")
                    vd_status = gr.Textbox(label="状态信息", lines=1)
                    
                    with gr.Accordion("下载成果物", open=False):
                        vd_out_wav = gr.File(label="音频文件 (WAV)")
                        vd_out_custom_srt = gr.File(label="句子级字幕 (SRT)")
                        vd_out_word_srt = gr.File(label="词级字幕 (SRT)")
                        vd_out_shorts_srt = gr.File(label="短视频格式字幕 (SRT)")

            def _build_instruct(groups):
                # 将中文选项映射回模型理解的英文指令
                selected = []
                for g in groups:
                    if g and g != "自动检测":
                        # 先尝试从方言映射表找，找不到再从通用映射表找，最后原样输出
                        val = INSTRUCT_MAP.get(g, g)
                        selected.append(val)
                if not selected: return None
                return ", ".join(selected)

            def _design_fn(text, lang, want_subs, ns, gs, dn, sp, du, pp, po, *groups):
                instruct = _build_instruct(groups)
                res = _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")
                if res[0] is None: return None, res[1], None, None, None, None
                
                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vd_btn.click(
                _design_fn,
                inputs=[vd_text, vd_lang, vd_want_subs, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                outputs=[vd_audio, vd_status, vd_out_wav, vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt],
            )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)