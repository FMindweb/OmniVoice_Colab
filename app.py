import os
import sys
import logging
import re
import uuid
import tempfile
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile

# ---------------------------------------------------------------------------
# 1. 路径与环境配置
# ---------------------------------------------------------------------------
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)

try:
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
    from subtitle import subtitle_maker
except ImportError:
    print("错误：请检查 OmniVoice 路径和依赖安装。")

temp_audio_dir = "./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. 模型加载
# ---------------------------------------------------------------------------
print("正在加载 OmniVoice 模型...")
try:
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )
except:
    model = OmniVoice.from_pretrained(
        "./OmniVoice_Model", 
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )

sampling_rate = model.sampling_rate
print("模型加载成功！")

# ---------------------------------------------------------------------------
# 3. 辅助配置
# ---------------------------------------------------------------------------
EVENT_TAGS = ["[笑声]", "[叹气]", "[确认-英]", "[疑问-英]", "[疑问-啊]", "[疑问-哦]", "[惊讶-啊]", "[惊讶-哇]", "[不满-唔]"]
_ALL_LANGUAGES = ["自动检测"] + sorted(lang_display_name(n) for n in LANG_NAMES)

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('.chat-input textarea');
    const tag_mapping = {"[笑声]": "[laughter]", "[叹气]": "[sigh]", "[确认-英]": "[confirmation-en]", "[疑问-英]": "[question-en]"};
    const real_tag = tag_mapping[tag_val] || tag_val;
    if (!textarea) return current_text + " " + real_tag;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    return current_text.slice(0, start) + " " + real_tag + " " + current_text.slice(end);
}
"""

# ---------------------------------------------------------------------------
# 4. 核心逻辑：生成与字幕
# ---------------------------------------------------------------------------
def _gen_single_sentence(text, language, ref_audio, instruct, config_params):
    gen_config = OmniVoiceGenerationConfig(
        num_step=config_params['ns'],
        guidance_scale=config_params['gs'],
        denoise=config_params['dn'],
        preprocess_prompt=config_params['pp'],
        postprocess_output=config_params['po'],
    )
    kw = dict(text=text.strip(), language=None if language == "自动检测" else language, generation_config=gen_config)
    if config_params['sp'] != 1.0: kw["speed"] = config_params['sp']
    if ref_audio:
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio)
    elif instruct:
        kw["instruct"] = instruct
    
    audio = model.generate(**kw)
    return (audio[0].squeeze(0).numpy() * 32767).astype(np.int16)

def multi_role_process(script, lang, role_list, global_ref, want_subs, ns, gs, dn, sp, pp, po):
    if not script.strip(): return None, "请输入剧本。", None, None, None, None

    roles_dict = {r["name"]: {"is_clone": r["is_clone"], "instruct": r["instruct"]} for r in role_list}
    lines = script.strip().split('\n')
    full_audio = []
    params = {'ns': ns, 'gs': gs, 'dn': dn, 'sp': sp, 'pp': pp, 'po': po}

    try:
        for line in lines:
            if "：" not in line and ":" not in line: continue
            role_name, content = re.split('：|:', line, 1)
            role_name = role_name.strip()
            if role_name not in roles_dict: continue
            
            role_cfg = roles_dict[role_name]
            ref_aud = global_ref if (role_cfg["is_clone"] and global_ref) else None
            instr = None if ref_aud else role_cfg["instruct"]
            
            wave = _gen_single_sentence(content, lang, ref_aud, instr, params)
            full_audio.append(wave)
            full_audio.append(np.zeros(int(sampling_rate * 0.5), dtype=np.int16)) # 停顿

        if not full_audio: return None, "匹配失败。", None, None, None, None

        # 保存音频
        final_wave = np.concatenate(full_audio)
        out_wav = f"{temp_audio_dir}/multi_{uuid.uuid4().hex[:8]}.wav"
        wavfile.write(out_wav, sampling_rate, final_wave)

        # 生成字幕
        c_srt, w_srt, s_srt = None, None, None
        if want_subs:
            try:
                whisper_lang = lang if lang != "自动检测" else None
                sub_res = subtitle_maker(out_wav, whisper_lang)
                if sub_res:
                    c_srt, w_srt, s_srt = sub_res[1], sub_res[2], sub_res[3]
            except Exception as e:
                logging.error(f"字幕生成失败: {e}")

        return (sampling_rate, final_wave), "合成成功！", out_wav, c_srt, w_srt, s_srt

    except Exception as e:
        return None, f"错误: {str(e)}", None, None, None, None

# ---------------------------------------------------------------------------
# 5. UI 构建
# ---------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet")) as demo:
    gr.Markdown("# 🎭 OmniVoice 多人对话合成 (含字幕版)")
    role_data = gr.State([{"name": "角色A", "instruct": "Male, Young", "is_clone": False}])

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("🔘 全局克隆音源", open=False):
                global_ref_audio = gr.Audio(label="克隆声音文件", type="filepath")

            @gr.render(inputs=role_data)
            def render_roles(data_list):
                for i, role in enumerate(data_list):
                    with gr.Container(variant="panel"):
                        with gr.Row():
                            name = gr.Textbox(value=role["name"], label="角色名", scale=2)
                            is_clone = gr.Checkbox(value=role["is_clone"], label="使用克隆", scale=1)
                            del_btn = gr.Button("🗑️", scale=0)
                        instr = gr.Textbox(value=role["instruct"], label="指令", visible=not role["is_clone"])

                        def update(n, c, ins, idx=i):
                            new_data = list(data_list)
                            if c: # 唯一克隆
                                for j in range(len(new_data)): new_data[j]["is_clone"] = False
                            new_data[idx] = {"name": n, "instruct": ins, "is_clone": c}
                            return new_data

                        name.change(update, [name, is_clone, instr], [role_data])
                        is_clone.change(update, [name, is_clone, instr], [role_data])
                        instr.change(update, [name, is_clone, instr], [role_data])
                        del_btn.click(lambda idx=i: [d for j, d in enumerate(data_list) if j != idx], None, [role_data])

            gr.Button("➕ 添加角色").click(lambda d: d + [{"name": f"角色{len(d)+1}", "instruct": "Female, Young", "is_clone": False}], [role_data], [role_data])
            
            script_input = gr.Textbox(label="剧本", lines=6, elem_classes="chat-input", placeholder="角色A：你好！")
            with gr.Row():
                for tag in EVENT_TAGS:
                    gr.Button(tag, size="sm").click(None, [gr.State(tag), script_input], script_input, js=INSERT_TAG_JS)
            
            lang_sel = _lang_dropdown()
            want_subs = gr.Checkbox(label="同步生成字幕文件 (Whisper)", value=True)
            
            with gr.Accordion("高级参数", open=False):
                ns = gr.Slider(4, 64, value=32, label="步数")
                gs = gr.Slider(0, 4, value=2.0, label="CFG")
                sp = gr.Slider(0.5, 1.5, value=1.0, label="语速")
                dn, pp, po = gr.Checkbox(label="去噪", value=True), gr.Checkbox(label="预处理", value=True), gr.Checkbox(label="后处理", value=True)

            run_btn = gr.Button("🚀 开始合成", variant="primary")

        with gr.Column(scale=1):
            audio_out = gr.Audio(label="对话预览")
            status_out = gr.Textbox(label="状态")
            with gr.Accordion("成果下载", open=True):
                file_wav = gr.File(label="音频 (WAV)")
                file_srt = gr.File(label="句子级字幕 (SRT)")
                file_word = gr.File(label="词级字幕 (SRT)")
                file_shorts = gr.File(label="短视频字幕 (SRT)")

    run_btn.click(
        multi_role_process,
        inputs=[script_input, lang_sel, role_data, global_ref_audio, want_subs, ns, gs, dn, sp, pp, po],
        outputs=[audio_out, status_out, file_wav, file_srt, file_word, file_shorts]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)