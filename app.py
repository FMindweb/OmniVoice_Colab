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
# Setup path to import subtitle_maker from /content/omnivoice-colab/OmniVoice/
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)
from subtitle import subtitle_maker

# Attempt to import Whisper's supported language dict to filter unsupported languages
try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Model Loading (Global Scope)
# ---------------------------------------------------------------------------
print("Loading model from k2-fsa/OmniVoice to cuda ...")

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
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Event Tags & JS Functions
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", 
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", 
    "[dissatisfaction-hnn]"
]

def create_js_hook(elem_id):
    return f"""
    (tag_val, current_text) => {{
        const textarea = document.querySelector('#{elem_id} textarea');
        if (!textarea) return current_text + " " + tag_val;
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        let prefix = " ";
        let suffix = " ";
        if (!current_text) return tag_val;
        if (start === 0) prefix = "";
        else if (current_text[start - 1] === ' ') prefix = "";
        if (end < current_text.length && current_text[end] === ' ') suffix = "";
        return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
    }}
    """

INSERT_TAG_JS_VC = create_js_hook("vc_textbox")
INSERT_TAG_JS_VD = create_js_hook("vd_textbox")
INSERT_TAG_JS_MD = create_js_hook("md_textbox") # Hook for Multi-Speaker Dialog

# ---------------------------------------------------------------------------
# UI Configurations & Language Mappings (Synced with OmniVoice Docs)
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

# Based exactly on https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md
_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
    "Accent/Dialect": [
        "American Accent", "Australian Accent", "British Accent", "Canadian Accent", 
        "Chinese Accent", "Indian Accent", "Korean Accent", "Portuguese Accent",
        "Russian Accent", "Japanese Accent",
        "Henan Dialect", "Shaanxi Dialect", "Sichuan Dialect", "Guizhou Dialect",
        "Yunnan Dialect", "Guilin Dialect", "Jinan Dialect", "Shijiazhuang Dialect",
        "Gansu Dialect", "Ningxia Dialect", "Qingdao Dialect", "Northeast Dialect"
    ]
}

# ---------------------------------------------------------------------------
# Core Logic & Helpers
# ---------------------------------------------------------------------------
def _is_whisper_supported(lang):
    if not lang or lang == "Auto":
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
    """Generates Subtitles only if user requested them and language is supported."""
    if not want_subs:
        return None, None, None

    if not _is_whisper_supported(lang):
        logging.warning(f"Language '{lang}' is likely unsupported by Whisper.")
        return None, None, None

    try:
        whisper_lang = lang if (lang and lang != "Auto") else None
        
        # --- 🚀 [HOTFIX] Bypass cuDNN version mismatch in Colab ---
        import torch
        original_cudnn_state = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False  # 临时禁用 cuDNN
        
        # 调用底层 Whisper 生成字幕
        whisper_results = subtitle_maker(wav_path, whisper_lang)
        
        # 恢复 cuDNN 状态，以免影响后续的 TTS 语音生成速度
        torch.backends.cudnn.enabled = original_cudnn_state
        # ----------------------------------------------------------

        if whisper_results and len(whisper_results) > 3:
            # 返回 SRT 路径
            return whisper_results[1], whisper_results[2], whisper_results[3] 
            
    except Exception as e:
        # 如果还有其他错误，至少恢复 cuDNN
        import torch
        torch.backends.cudnn.enabled = True
        logging.warning(f"Subtitle generation failed: {e}")

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
        return None, "Please enter the text to synthesize."

    if mode == "clone" and ref_audio and not ref_text:
        try:
            whisper_lang = language if (language and language != "Auto") else None
            whisper_results = subtitle_maker(ref_audio, whisper_lang)
            if whisper_results and len(whisper_results) > 7:
                ref_text = whisper_results[7]
        except Exception as e:
            logging.warning(f"Fallback transcription failed: {e}")

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)

    if speed is not None and float(speed) != 1.0: kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0: kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio: return None, "Please upload a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if mode == "design":
        if instruct and instruct.strip(): kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    waveform = (audio[0] * 32767).astype(np.int16)
    
    return (sampling_rate, waveform), "Done."

def _build_instruct_str(gender, age, pitch, style, accent):
    """Utility to combine selected dropdowns into an instruct string"""
    selections = [gender, age, pitch, style, accent]
    valid = [s for s in selections if s and s != "Auto"]
    return ", ".join(valid)

# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}
.tag-container { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; margin-top: 5px !important; margin-bottom: 10px !important; border: none !important; background: transparent !important;}
.tag-btn { min-width: fit-content !important; width: auto !important; height: 32px !important; font-size: 13px !important; background: #eef2ff !important; border: 1px solid #c7d2fe !important; color: #3730a3 !important; border-radius: 6px !important; padding: 0 10px !important; margin: 0 !important; box-shadow: none !important;}
.tag-btn:hover { background: #c7d2fe !important; transform: translateY(-1px); }
"""

def _lang_dropdown(label="Language (optional)", value="Auto"):
    return gr.Dropdown(label=label, choices=_ALL_LANGUAGES, value=value, interactive=True)

def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
        du = gr.Number(value=None, label="Duration (seconds)")
        ns = gr.Slider(4, 64, value=32, step=1, label="Inference Steps")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
        pp = gr.Checkbox(label="Preprocess Prompt", value=True)
        po = gr.Checkbox(label="Postprocess Output", value=True)
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
        # 1. Voice Clone Tab
        # ==============================================================
        with gr.TabItem("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(label="Text to Synthesize", lines=4, elem_id="vc_textbox")
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(fn=None, inputs=[btn, vc_text], outputs=vc_text, js=INSERT_TAG_JS_VC)
                    with gr.Row():
                      vc_lang = _lang_dropdown()
                      vc_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vc_ref_audio = gr.Audio(label="Reference Audio", type="filepath", elem_classes="compact-audio")
                    vc_ref_text = gr.Textbox(label="Reference Text", lines=2)
                    vc_btn = gr.Button("Generate", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="Output Audio", type="numpy")
                    vc_status = gr.Textbox(label="Status", lines=1)
                    with gr.Accordion("Download files", open=False):
                        vc_out_wav, vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt = gr.File(), gr.File(), gr.File(), gr.File()

            def _clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po):
                res = _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text)
                if res[0] is None: return None, res[1], None, None, None, None
                sr, waveform = res[0]
                tmp_wav=tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return res[0], res[1], tmp_wav, c_srt, w_srt, s_srt

            vc_btn.click(_clone_fn, inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po], outputs=[vc_audio, vc_status, vc_out_wav, vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt])

        # ==============================================================
        # 2. Voice Design Tab
        # ==============================================================
        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(label="Text to Synthesize", lines=4, elem_id="vd_textbox")
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(fn=None, inputs=[btn, vd_text], outputs=vd_text, js=INSERT_TAG_JS_VD)

                    with gr.Row():
                      vd_lang = _lang_dropdown(value='Auto')
                      vd_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vd_btn = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Character Voice Design Parameters", open=False):
                        vd_gender = gr.Dropdown(label="Gender", choices=["Auto"] + _CATEGORIES["Gender"], value="Female")
                        vd_age = gr.Dropdown(label="Age", choices=["Auto"] + _CATEGORIES["Age"], value="Young Adult")
                        vd_pitch = gr.Dropdown(label="Pitch", choices=["Auto"] + _CATEGORIES["Pitch"], value="Auto")
                        vd_style = gr.Dropdown(label="Style", choices=["Auto"] + _CATEGORIES["Style"], value="Auto")
                        vd_accent = gr.Dropdown(label="Accent / Dialect", choices=["Auto"] + _CATEGORIES["Accent/Dialect"], value="Auto")
                        
                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="Output Audio", type="numpy")
                    vd_status = gr.Textbox(label="Status", lines=1)
                    with gr.Accordion("Download files", open=False):
                        vd_out_wav, vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt = gr.File(), gr.File(), gr.File(), gr.File()

            def _design_fn(text, lang, want_subs, ns, gs, dn, sp, du, pp, po, gen, age, pit, sty, acc):
                instruct = _build_instruct_str(gen, age, pit, sty, acc)
                res = _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")
                if res[0] is None: return None, res[1], None, None, None, None
                sr, waveform = res[0]
                tmp_wav=tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return res[0], res[1], tmp_wav, c_srt, w_srt, s_srt

            vd_btn.click(_design_fn, inputs=[vd_text, vd_lang, vd_want_subs, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po, vd_gender, vd_age, vd_pitch, vd_style, vd_accent], outputs=[vd_audio, vd_status, vd_out_wav, vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt])

        # ==============================================================
        # 3. Multi-Speaker Dialogue Tab (Upgraded Options)
        # ==============================================================
        with gr.TabItem("Multi-Speaker Dialogue"):
            gr.Markdown("Define roles using **Voice Design Dropdowns** or Audio Clone, and synthesize a multi-character script.")
            role_inputs = [] # Store references to role UI elements (8 items per role)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Role Definitions")
                    
                    # Pre-defined defaults for 4 slots [Name, Mode, Gender, Age, Pitch, Style, Accent]
                    defaults = [
                        ("Narrator", "Design", "Male", "Middle-aged", "Low Pitch", "Auto", "Auto"),
                        ("Alice", "Design", "Female", "Young Adult", "High Pitch", "Auto", "Auto"),
                        ("Bob", "Design", "Male", "Young Adult", "Moderate Pitch", "Auto", "Auto"),
                        ("System", "Design", "Female", "Auto", "Auto", "Whisper", "Auto")
                    ]
                    
                    for i in range(4):
                        d_name, d_mode, d_gen, d_age, d_pit, d_sty, d_acc = defaults[i]
                        with gr.Accordion(f"Role {i+1} Configuration", open=(i<2)):
                            with gr.Row():
                                r_name = gr.Textbox(label="Character Name", value=d_name, scale=2)
                                r_mode = gr.Radio(["Design", "Clone"], value=d_mode, label="Mode", scale=1)
                            
                            # Interactive Options for Design Mode
                            with gr.Group(visible=(d_mode=="Design")) as r_design_group:
                                with gr.Row():
                                    r_gender = gr.Dropdown(label="Gender", choices=["Auto"] + _CATEGORIES["Gender"], value=d_gen)
                                    r_age = gr.Dropdown(label="Age", choices=["Auto"] + _CATEGORIES["Age"], value=d_age)
                                    r_pitch = gr.Dropdown(label="Pitch", choices=["Auto"] + _CATEGORIES["Pitch"], value=d_pit)
                                with gr.Row():
                                    r_style = gr.Dropdown(label="Style", choices=["Auto"] + _CATEGORIES["Style"], value=d_sty)
                                    r_accent = gr.Dropdown(label="Accent/Dialect", choices=["Auto"] + _CATEGORIES["Accent/Dialect"], value=d_acc)
                            
                            # Audio Input for Clone Mode
                            r_ref = gr.Audio(label="Reference Audio", type="filepath", visible=(d_mode=="Clone"), elem_classes="compact-audio")
                            
                            def toggle_role_mode(m):
                                return gr.update(visible=(m=="Design")), gr.update(visible=(m=="Clone"))
                            r_mode.change(toggle_role_mode, inputs=r_mode, outputs=[r_design_group, r_ref])
                            
                            # Audition Feature
                            with gr.Row():
                                r_prev_btn = gr.Button("🎧 Preview Voice", size="sm")
                                r_prev_out = gr.Audio(label="", elem_classes="compact-audio", show_label=False)
                            
                            def preview_voice(mode, gen, age, pit, sty, acc, ref):
                                test_txt = "Hello, this is a quick test of my voice."
                                m = mode.lower()
                                inst = _build_instruct_str(gen, age, pit, sty, acc) if m == "design" else None
                                res = _gen_core(test_txt, "Auto", ref, inst, 32, 2.0, True, 1.0, None, True, True, mode=m)
                                return res[0] if res[0] else None
                                
                            r_prev_btn.click(preview_voice, inputs=[r_mode, r_gender, r_age, r_pitch, r_style, r_accent, r_ref], outputs=[r_prev_out])
                            
                            # Append to main list (8 items per role)
                            role_inputs.extend([r_name, r_mode, r_gender, r_age, r_pitch, r_style, r_accent, r_ref])
                            
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Dialogue Script")
                    md_text = gr.Textbox(
                        label="Script format: 'Name: Text'", lines=10, elem_id="md_textbox",
                        placeholder="Narrator: Once upon a time...\nAlice: [laughter] Hello Bob!\nBob: Hi Alice!"
                    )
                    
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(fn=None, inputs=[btn, md_text], outputs=md_text, js=INSERT_TAG_JS_MD)
                            
                    with gr.Row():
                        md_lang = _lang_dropdown()
                        md_want_subs = gr.Checkbox(label="Generate Subtitles?", value=True)
                        
                    md_ns, md_gs, md_dn, md_sp, md_du, md_pp, md_po = _gen_settings()
                    md_btn = gr.Button("🚀 Synthesize Full Script", variant="primary", size="lg")
                    
                    gr.Markdown("### 3. Result")
                    md_audio = gr.Audio(label="Final Audio Output")
                    md_status = gr.Textbox(label="Status")
                    with gr.Accordion("Download files", open=True):
                        md_out_wav = gr.File(label="Generated Audio (WAV)")
                        md_out_srt = gr.File(label="Sentence Level SRT")

            def synthesize_script(script, lang, subs, ns, gs, dn, sp, du, pp, po, *roles_data):
                if not script.strip(): return None, "Script is empty.", None, None
                
                roles = {}
                # Extract 8 properties per role
                for i in range(4):
                    idx = i * 8
                    r_n = roles_data[idx].strip()
                    r_m = roles_data[idx+1].lower()
                    r_g = roles_data[idx+2]
                    r_a = roles_data[idx+3]
                    r_p = roles_data[idx+4]
                    r_s = roles_data[idx+5]
                    r_ac = roles_data[idx+6]
                    r_ref = roles_data[idx+7]
                    
                    if r_n: 
                        instruct = _build_instruct_str(r_g, r_a, r_p, r_s, r_ac)
                        roles[r_n] = {"mode": r_m, "instruct": instruct, "ref": r_ref}
                
                lines = script.strip().split('\n')
                audio_segments = []
                
                for line in lines:
                    if ":" not in line and "：" not in line: continue
                    parts = re.split(r'[:：]', line, 1)
                    char_name = parts[0].strip()
                    content = parts[1].strip()
                    
                    if char_name not in roles or not content: continue
                        
                    role = roles[char_name]
                    res = _gen_core(
                        text=content, language=lang, 
                        ref_audio=role["ref"], instruct=role["instruct"], 
                        num_step=ns, guidance_scale=gs, denoise=dn, 
                        speed=sp, duration=du, preprocess_prompt=pp, postprocess_output=po, 
                        mode=role["mode"]
                    )
                    
                    if res[0] is not None:
                        sr, wave = res[0]
                        audio_segments.append(wave)
                        audio_segments.append(np.zeros(int(sr * 0.5), dtype=np.int16)) # 0.5s silence
                
                if not audio_segments: return None, "No valid characters matched the script.", None, None
                
                final_wave = np.concatenate(audio_segments)
                tmp_wav = tts_file_name("dialogue", language=lang)
                wavfile.write(tmp_wav, sampling_rate, final_wave)
                
                c_srt, w_srt, s_srt= generate_subtitles_if_needed(tmp_wav, lang, subs)
                return (sampling_rate, final_wave), "Dialogue Synthesized Successfully!", tmp_wav, c_srt

            md_inputs = [md_text, md_lang, md_want_subs, md_ns, md_gs, md_dn, md_sp, md_du, md_pp, md_po] + role_inputs
            md_btn.click(synthesize_script, inputs=md_inputs, outputs=[md_audio, md_status, md_out_wav, md_out_srt])
        
        # ==============================================================
        # 5. SRT to Speech Tab (SRT 转语音)
        # ==============================================================
        with gr.TabItem("SRT to Speech (字幕转语音)"):
            gr.Markdown("上传 `.srt` 字幕文件，模型会自动提取文本，并使用你配置的声音将其转换为完整的配音音频。")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 上传字幕 & 配置声音")
                    srt_file = gr.File(label="上传 .srt 文件", file_types=[".srt"])
                    
                    with gr.Accordion("声音配置 (Voice Configuration)", open=True):
                        srt_mode = gr.Radio(["Design", "Clone"], value="Design", label="声音模式 (Mode)")
                        
                        # Design 模式配置
                        with gr.Group(visible=True) as srt_design_group:
                            with gr.Row():
                                srt_gender = gr.Dropdown(label="Gender", choices=["Auto"] + _CATEGORIES["Gender"], value="Female")
                                srt_age = gr.Dropdown(label="Age", choices=["Auto"] + _CATEGORIES["Age"], value="Young Adult")
                                srt_pitch = gr.Dropdown(label="Pitch", choices=["Auto"] + _CATEGORIES["Pitch"], value="Auto")
                            with gr.Row():
                                srt_style = gr.Dropdown(label="Style", choices=["Auto"] + _CATEGORIES["Style"], value="Auto")
                                srt_accent = gr.Dropdown(label="Accent/Dialect", choices=["Auto"] + _CATEGORIES["Accent/Dialect"], value="Auto")
                        
                        # Clone 模式配置
                        srt_ref = gr.Audio(label="上传参考克隆音频", type="filepath", visible=False, elem_classes="compact-audio")
                        
                        def toggle_srt_mode(m):
                            return gr.update(visible=(m=="Design")), gr.update(visible=(m=="Clone"))
                        srt_mode.change(toggle_srt_mode, inputs=srt_mode, outputs=[srt_design_group, srt_ref])
                    
                    srt_lang = _lang_dropdown()
                    # 复用现有的高级设置
                    srt_ns, srt_gs, srt_dn, srt_sp, srt_du, srt_pp, srt_po = _gen_settings()
                    
                    srt_btn = gr.Button("🚀 转换 SRT 为语音", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 2. 生成结果")
                    srt_audio_out = gr.Audio(label="最终配音音频")
                    srt_status = gr.Textbox(label="状态信息")
                    srt_out_wav = gr.File(label="下载完整音频 (WAV)")
                    srt_parsed_text = gr.Textbox(label="解析出的文本 (仅供核对)", lines=8)

            # --- SRT 解析与生成逻辑 ---
            def process_srt_to_speech(file_obj, lang, ns, gs, dn, sp, du, pp, po, mode, gen, age, pit, sty, acc, ref):
                if not file_obj:
                    return None, "请上传一个 SRT 文件。", None, ""
                
                # 定义一个内部辅助函数：将 SRT 的 00:00:01,500 转换成秒数
                def srt_time_to_seconds(time_str):
                    hrs, mins, secs = time_str.replace(',', '.').split(':')
                    return int(hrs) * 3600 + int(mins) * 60 + float(secs)

                # 1. 解析 SRT 文件，提取【开始时间】、【结束时间】和【文本】
                try:
                    with open(file_obj.name, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # 使用正则精准匹配 SRT 格式
                    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n$|$)', re.DOTALL)
                    matches = pattern.findall(content)
                    
                    srt_data = []
                    for m in matches:
                        # 过滤 HTML 标签并清理换行
                        clean_text = re.sub(r'<[^>]+>', '', m[3].replace('\n', ' ').strip())
                        if clean_text:
                            srt_data.append({
                                "start": srt_time_to_seconds(m[1]),
                                "end": srt_time_to_seconds(m[2]),
                                "text": clean_text
                            })
                except Exception as e:
                    return None, f"解析 SRT 文件失败: {str(e)}", None, ""
                
                if not srt_data:
                    return None, "未能从文件中提取出有效文本，请检查 SRT 格式。", None, ""
                    
                parsed_preview = "\n".join([x["text"] for x in srt_data])
                
                # 2. 准备模型指令
                instruct = _build_instruct_str(gen, age, pit, sty, acc) if mode.lower() == "design" else None
                audio_segments = []
                current_audio_time = 0.0  # 用于追踪当前合成音频在时间轴上的绝对位置
                
                # 3. 逐句精确合成音频并对齐
                for item in srt_data:
                    target_start = item["start"]
                    
                    # 计算需要补齐的静音差值
                    wait_duration = target_start - current_audio_time
                    
                    # 只有当下一句的开始时间大于当前时间时，才插入静音补齐
                    if wait_duration > 0:
                        silence = np.zeros(int(sampling_rate * wait_duration), dtype=np.int16)
                        audio_segments.append(silence)
                        current_audio_time += wait_duration
                    
                    # 核心生成 (完全保留了你的所有原版参数)
                    res = _gen_core(
                        text=item["text"], language=lang, 
                        ref_audio=ref, instruct=instruct, 
                        num_step=ns, guidance_scale=gs, denoise=dn, 
                        speed=sp, duration=du, preprocess_prompt=pp, postprocess_output=po, 
                        mode=mode.lower()
                    )
                    
                    if res[0] is not None:
                        sr, wave = res[0]
                        audio_segments.append(wave)
                        # 更新当前时间轴：加上刚生成这句话的真实真实长度
                        current_audio_time += (len(wave) / sr)
                
                if not audio_segments:
                    return None, "生成失败，可能是参数设置问题。", None, parsed_preview
                
                # 4. 拼接并导出
                final_wave = np.concatenate(audio_segments)
                tmp_wav = tts_file_name("precision_srt", language=lang)
                wavfile.write(tmp_wav, sampling_rate, final_wave)
                
                return (sampling_rate, final_wave), "SRT 转换配音成功！", tmp_wav, parsed_preview
                
                # 4. 拼接并保存最终音频
                final_wave = np.concatenate(audio_segments)
                tmp_wav = tts_file_name("srt_dubbing", language=lang)
                wavfile.write(tmp_wav, sampling_rate, final_wave)
                
                return (sampling_rate, final_wave), "SRT 转换配音成功！", tmp_wav, parsed_preview

            # 绑定按钮事件
            srt_inputs = [srt_file, srt_lang, srt_ns, srt_gs, srt_dn, srt_sp, srt_du, srt_pp, srt_po, 
                          srt_mode, srt_gender, srt_age, srt_pitch, srt_style, srt_accent, srt_ref]
            srt_btn.click(
                process_srt_to_speech,
                inputs=srt_inputs,
                outputs=[srt_audio_out, srt_status, srt_out_wav, srt_parsed_text]
            )

        # ==============================================================
        # 4. Voice Design Dictionary / Reference Tab (NEW)
        # ==============================================================
        with gr.TabItem("📖 Voice Design Reference"):
            gr.Markdown("""
            ### 声音设计参数速查表 (Voice Design Parameters Guide)
            以下数据源于 OmniVoice 官方 GitHub 仓库：[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md)
            
            你可以将这些参数作为 `instruct` 文本框的输入，或者在“多人对话”与“人声设计”选项卡的下拉菜单中直接选择它们。
            模型会自动理解这些维度的组合（例如：`Female, Young Adult, High Pitch, Whisper`）。
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    #### 1. 核心维度 (Core Attributes)
                    | 类别 (Category) | 支持的参数值 (Supported Values) | 中文释义 |
                    | :--- | :--- | :--- |
                    | **Gender (性别)** | `Male`, `Female` | 男, 女 |
                    | **Age (年龄)** | `Child`, `Teenager`, `Young Adult`, `Middle-aged`, `Elderly` | 儿童, 少年, 青年, 中年, 老年 |
                    | **Pitch (音高)** | `Very Low Pitch`, `Low Pitch`, `Moderate Pitch`, `High Pitch`, `Very High Pitch` | 极低音, 低音, 中音, 高音, 极高音 |
                    | **Style (风格)** | `Whisper` | 耳语/悄悄话 |
                    """)
                with gr.Column():
                    gr.Markdown("""
                    #### 2. 口音与方言 (Accents & Dialects)
                    | 类别 (Category) | 支持的参数值 (Supported Values) | 中文释义 |
                    | :--- | :--- | :--- |
                    | **English Accents (英语口音)** | `American Accent`, `British Accent`, `Australian Accent`, `Canadian Accent`, `Indian Accent`, `Chinese Accent`, `Korean Accent`, `Portuguese Accent`, `Russian Accent`, `Japanese Accent` | 美音, 英音, 澳洲口音, 加拿大口音, 印度口音, 中式英语, 韩式英语, 葡萄牙口音, 俄式口音, 日式英语 |
                    | **Chinese Dialects (中文方言)** | `Henan Dialect`, `Shaanxi Dialect`, `Sichuan Dialect`, `Guizhou Dialect`, `Yunnan Dialect`, `Guilin Dialect`, `Jinan Dialect`, `Shijiazhuang Dialect`, `Gansu Dialect`, `Ningxia Dialect`, `Qingdao Dialect`, `Northeast Dialect` | 河南话, 陕西话, 四川话, 贵州话, 云南话, 桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话, 青岛话, 东北话 |
                    """)
            gr.Markdown("""
            *💡 **提示**: 你可以在剧本的文本中直接插入情感语气词，例如：`[laughter]` (笑声), `[sigh]` (叹气), `[surprise-wa]` (哇), `[dissatisfaction-hnn]` (不满的嗯声) 等，提升角色对话的生动程度。*
            """)

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)