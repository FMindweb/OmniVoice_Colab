# 最强 AI 语音克隆 & 极速 TTS 工具箱

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IkNJKU4xQQakRby0RjJ7-ZGq7NVS1Hno?usp=sharing)
[![Blog Post](https://img.shields.io/badge/Blog-详细教程-orange)](https://fm192.blogspot.com/2026/04/google-colab-ai-11.html)
[![YouTube](https://img.shields.io/badge/YouTube-FM19.2-red)](https://www.youtube.com/@fm19.2?sub_confirmation=1)

本工具箱基于 **Google Colab 免费 T4 算力**，集成了深度优化的 TTS 引擎与人性化的 Web UI。旨在解决目前 AI 配音中存在的**合成速度慢、语种覆盖少、多音字不准**以及**解说视频字幕对齐难**等核心痛点。

---

## ✨ 核心特性

* **⚡ 1:1 极速合成**：性能极致优化，45 秒音频仅需约 40 秒完成推理，真正实现实时响应。
* **👥 多角色对话**：支持最多 **4 个角色** 同台对话。通过 `1: 文本` 或 `2: 文本` 的简单指令即可模拟多人剧本。
* **⏱️ SRT 字幕精准转语音**：上传 SRT 文件，系统自动根据毫秒级时间戳填充语音，输出音频与字幕完美对齐。
* **🎨 深度语音设计 (Voice Design)**：支持自定义年龄、性别、音调、风格（如轻声细语），捏出专属人声。
* **🔡 拼音音标修正**：支持 **拼音 + 音调** 输入（例如：`zhong4 xia4 zhong3 zi3`），彻底解决多音字发音错误。
* **🌍 600+ 语种与方言**：涵盖全球主流语言及方言（河南话、四川话、粤语、甚至各种地方口音）。

---

## 🛠️ 快速上手

1.  **启动脚本**：点击顶部的 `Open In Colab` 图标。
2.  **保存副本**：在 Colab 菜单栏选择 `File (文件)` -> `Save a copy in Drive (在云端硬盘中保存副本)`。
3.  **配置环境**：
    * 点击 `Runtime (运行时)` -> `Change runtime type (更改运行时类型)`。
    * **Hardware accelerator (硬件加速器)** 务必选择 **T4 GPU**。
4.  **一键部署**：点击 `Runtime (运行时)` -> `Run all (全部运行)`。
5.  **进入界面**：等待约 2 分钟，点击输出结果中的 `public URL` (Gradio 链接) 即可开始创作。

---

Based on **Google Colab's free computing power**, this toolbox integrates a deeply optimized TTS engine and Web UI. It is designed to solve common issues in voice cloning such as **slow inference, limited language support, inaccurate polyphone pronunciation, and subtitle misalignment.**

### ✨ Key Features
* **⚡ 1:1 Real-time Synthesis:** Optimized performance allows for near 1:1 inference speed (e.g., 45s audio in ~40s).
* **👥 Multi-Character Dialogue:** Supports up to **4 characters** in a single session. Use simple markers (e.g., `1: text`) to simulate scripts.
* **⏱️ Precise SRT Alignment:** Upload an SRT file, and the AI generates audio perfectly synced to the millisecond timestamps.
* **🎨 Advanced Voice Design:** Create unique voices by adjusting gender, age, pitch, and style (e.g., whispering).
* **🔡 Phonetic Precision:** Supports **Pinyin with tones** for Chinese and IPA for English to fix pronunciation errors.
* **🌍 600+ Languages & Accents:** Supports global languages and various regional Chinese dialects.

### 🛠️ Quick Start
1. **Open Colab:** Click the [Open In Colab] badge above.
2. **Save Copy:** Go to `File` -> `Save a copy in Drive` (**Required**).
3. **Enable T4 GPU:** `Runtime` -> `Change runtime type` -> Select `T4 GPU`.
4. **Run All:** Click `Runtime` -> `Run all`.
5. **Access UI:** Wait for setup (~2 min) and click the `public URL` provided in the output.

---

## 📖 教程与技术支持

* **详细文案版教程**：[FM19.2 Blog - 极速语音克隆指南](https://fm192.blogspot.com/2026/04/google-colab-ai-11.html)
* **视频实测演示**：前往 YouTube 频道 [FM19.2](https://www.youtube.com/@fm19.2?sub_confirmation=1) 观看本项目演示。

---

## ⚙️ 常见问题 (FAQ)

> **Q: 多人语音模式运行报错怎么办？**
>
> **A:** 请确保在进入“多人语音”标签页前，先在“语音克隆”或“语音设计”中成功运行过一次合成，以激活模型缓存。
>
> **Q: 为什么合成的语速飞快？**
>
> **A:** 如果你在界面设置了“指定秒数”，系统会忽略语速滑块，优先满足时间长度要求。若想手动控制语速，请清空时间限制框。

---

## 🤝 声明
本仓库代码及 Colab 脚本仅供学习与交流使用，请勿用于任何违反法律法规的行为。

**如果你觉得这个项目好用，欢迎给一个 ⭐ Star 鼓励，或者订阅我的频道获取更多白嫖 AI 技巧！**

---
*Created by [FM](https://www.youtube.com/@fm19.2?sub_confirmation=1)*
