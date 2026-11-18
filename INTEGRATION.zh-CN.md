# Confucius4-TTS 外部工程调用指南

面向需要把本项目当作“语音合成后端”接入的其他工程。三种接入方式按推荐顺序：

| 方式 | 适用场景 | 模型加载 | 首选场景 |
|---|---|---|---|
| **HTTP 服务**（`server.py`） | 跨机器 / 跨语言调用方 | 服务启动时一次 | 内网服务、远程调用 |
| **Python API**（进程内） | Python 工程、批处理脚本 | 自己控制，进程存活期间复用 | 批量合成、需要精细参数 |
| **CLI**（`example.py`） | Shell 脚本、一次性任务 | **每次调用都重新加载（约 1 分钟）** | 仅适合少量调用 |

> 本机基础环境（RTX 5090）已验证：conda 环境 `confuciustts`、模型在本地缓存，PyTorch 为 **cu128** 构建。不要在该环境混装 cu126 轮子。vLLM 后端需要按下文另行安装和验证，不属于基础环境已验证范围。
>
> 文中的 `HF_HUB_OFFLINE=1 python ...` 是 Bash / Git Bash 写法。PowerShell 请先执行 `$env:HF_HUB_OFFLINE = "1"`，再单独运行 `python ...`。

---

## 一、HTTP 服务

### 1. 环境与启动

`server.py` 使用 **vLLM 后端**（T2S 阶段加速），比基础环境多一层依赖：

```bash
# 首次：从基础环境克隆并加装 vLLM 依赖
conda create -n confuciustts_vllm --clone confuciustts -y
conda activate confuciustts_vllm
pip install -r requirements_vllm_add.txt   # vLLM 0.16.0 (V1 engine)

# 启动（模型已随仓库放在 models/hf/，通过 HF_HUB_CACHE 指向它；
# HF_HUB_OFFLINE=1 跳过联网检查）
cd /path/to/Confucius4-TTS
HF_HUB_CACHE="$PWD/models/hf" HF_HUB_OFFLINE=1 python server.py --port 8000
```

启动参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `config/inference_config.yaml` | 推理配置；默认值是相对路径，因此需从仓库根目录启动，或传入绝对路径 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--gpu_memory_utilization` | `0.4` | vLLM 显存占比；不是整个进程最终的显存占用保证 |

按上述 `python server.py` 方式启动时，模型在端口开放**之前**加载（约 1 分钟）。就绪后：

```text
GET /health → {"status": "ok", "sample_rate": 22050}
```

> **不要直接暴露到公网**：默认监听 `0.0.0.0`，两个合成接口均没有身份认证、TLS 或限流。生产使用时应放在可信内网，或由网关 / 反向代理补充认证、TLS、请求限流和超时控制。
>
> **Windows 已知坑**：`server.py` 把上传的参考音频写到 `Path("/tmp")`。原生 Windows Python 会将其解析为“当前工作目录所在驱动器”的根目录，例如从 E: 盘仓库启动时是 `E:\tmp`，不是固定为 `C:\tmp`。目录不存在时上传会失败。启动前可手动创建对应目录；修改代码时，可使用 `Path(tempfile.gettempdir())` 并补充 `import tempfile`。
>
> **无 vLLM 环境时**：当前 `server.py` 与 vLLM 的异步接口耦合，**不能只替换 import 并删除构造参数**。基础后端的 `generate()` 是同步方法，也没有 `generate_stream()`。没有 vLLM 时请直接使用下文的 Python API 或 CLI；若必须提供 HTTP，需要另行实现同步生成适配层，并禁用或单独实现流式接口。

### 2. 非流式接口 `POST /api/tts`

请求为 `multipart/form-data`：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | string | ✅ | 待合成文本，≤1024 字符；标点（！？…、）驱动语调 |
| `lang` | string | 否 | 语言码，默认 `zh`。可选：`zh en ja ko de fr th id vi es pt it ru ms`（14 种） |
| `reference` | file | ✅ | 参考音频，决定**音色+情绪+语速**，≤50MB。本机已验证 WAV / FLAC / MP3 / OGG；**M4A 请先转码**，见“参考音频规范” |

响应：`audio/wav`（PCM 16-bit，22050 Hz，单声道），元数据在响应头：

| Header | 含义 |
|---|---|
| `X-Sample-Rate` | 采样率（22050） |
| `X-Duration-Sec` | 音频时长 |
| `X-Elapsed-Sec` | 本次模型生成耗时，不含客户端上传时间 |

curl 示例：

```bash
curl -F "text=要合成的文本" -F "lang=zh" -F "reference=@ref.wav" \
     http://localhost:8000/api/tts -o out.wav
```

Python（requests）示例：

```python
import requests

with open("ref.wav", "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/tts",
        data={"text": "你好，世界。", "lang": "zh"},
        files={"reference": ("ref.wav", f, "audio/wav")},
        timeout=120,
    )
r.raise_for_status()
print(r.headers["X-Duration-Sec"], "seconds of audio")
with open("out.wav", "wb") as out:
    out.write(r.content)
```

### 3. 流式接口 `POST /api/tts/stream`

请求字段同上。响应为 **chunked 原始 PCM**，边生成边推：

| Header | 值 |
|---|---|
| `X-Sample-Rate` | `22050` |
| `X-Channels` | `1`（单声道） |
| `X-Encoding` | `int16-LE-PCM`（无 WAV 头，裸字节流） |

HTTP 分块不是音频帧边界：客户端必须跨分块保留未对齐的字节，并使用持续的音频输出流，不能假设每次 `iter_content()` 都返回完整的 int16 样本。

**错误协议限制**：合成开始后的异常会以内嵌文本 `\n[ERROR] ...` 追加到 HTTP 200 的 PCM 尾部。客户端必须检测并报告它，不能播放或简单当作“非二进制尾部”忽略。该协议无法彻底排除 PCM 数据碰巧包含相同标记的歧义；若能修改服务端，生产环境应改用带类型和长度的帧，或独立的错误通道。

下面的示例兼容当前服务端协议：保留跨块字节、检测跨块错误标记，并用 `RawOutputStream` 连续播放。它仍受上述标记歧义限制。

客户端需要额外安装依赖（服务端环境不需要安装 `sounddevice`）：

```bash
pip install requests sounddevice
```

```python
import requests
import sounddevice as sd

ERROR_MARK = b"\n[ERROR] "


def iter_pcm(response):
    """Yield aligned PCM bytes; raise instead of playing an inline error."""
    pending = b""
    chunks = response.iter_content(chunk_size=8192)
    for chunk in chunks:
        pending += chunk
        marker_at = pending.find(ERROR_MARK)
        if marker_at >= 0:
            # The marker and its message may arrive in separate HTTP chunks.
            error_bytes = pending[marker_at:] + b"".join(chunks)
            detail = error_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS stream failed: {detail}")

        # Hold back enough bytes to recognize a marker split across chunks.
        safe_len = max(0, len(pending) - (len(ERROR_MARK) - 1))
        safe_len -= safe_len % 2  # int16-LE requires complete two-byte samples.
        if safe_len:
            yield pending[:safe_len]
            pending = pending[safe_len:]

    marker_at = pending.find(ERROR_MARK)
    if marker_at >= 0:
        detail = pending[marker_at:].decode("utf-8", errors="replace")
        raise RuntimeError(f"TTS stream failed: {detail}")
    if len(pending) % 2:
        raise RuntimeError("TTS stream ended with an incomplete PCM sample")
    if pending:
        yield pending


with open("ref.wav", "rb") as f:
    with requests.post(
        "http://localhost:8000/api/tts/stream",
        data={"text": "你好", "lang": "zh"},
        files={"reference": ("ref.wav", f, "audio/wav")},
        stream=True,
        timeout=(10, 300),
    ) as r:
        r.raise_for_status()  # Handles validation errors before streaming starts.
        sample_rate = int(r.headers["X-Sample-Rate"])
        channels = int(r.headers["X-Channels"])
        with sd.RawOutputStream(
            samplerate=sample_rate, channels=channels, dtype="int16"
        ) as player:
            for pcm in iter_pcm(r):
                player.write(pcm)
```

### 4. 错误码

| HTTP | 场景 |
|---|---|
| 400 | 语言不支持 / text 为空或超 1024 字符 / 参考音频扩展名不在白名单 |
| 413 | 参考音频 > 50MB |
| 503 | HTTP 应用已建立但模型未就绪；按本文 `main()` 启动方式通常不可见，因为端口在模型加载后才开放 |
| 500 | 合成或音频解码失败。模型生成阶段的异常通常在 `detail` 中带异常类型与信息；上传落盘等其他异常不保证相同响应格式 |

流式响应开始后再发生合成异常时，HTTP 状态仍是 200，错误通过上述内嵌标记返回。

### 5. 并发与健康检查

- 服务只有一个 worker 和一个顶层模型对象，但**没有请求级互斥锁**；单 worker 不等于请求自动串行。vLLM 异步解码等待期间，其他请求可以进入生成流程，因此 GPU 任务可能重叠。
- 调用方若需控制显存和延迟，应**自行排队或限并发为 1**，不要依赖服务端隐式串行。多请求的等待时间也应计入客户端超时。
- W2V、S2A、BigVGAN 等阶段包含同步计算，会阻塞事件循环；合成期间第二个请求甚至 `/health` 都可能暂时无响应。健康检查超时不一定代表进程已退出，应结合进程状态和日志判断。
- 吞吐参考：本机 RTX 5090 + transformers 基础后端曾测得约 **1× 实时**（7 秒语音约 6.3 秒生成）。该数据不是 vLLM 或其他硬件的性能保证。

---

## 二、Python 进程内 API（推荐给 Python 工程）

基础环境 `confuciustts` 即可用。**模型只加载一次，进程内复用**。

当前版本的 `./checkpoints` tokenizer 和统计文件按**进程当前工作目录**解析，而不是按 `config_path` 解析。外部工程须先固定输入、输出的绝对路径，再切换到仓库根目录；`pip install -e .` 只能省略 `sys.path` 设置，不能省略这个目录前提。

```python
import os
import sys
from pathlib import Path

REPO_ROOT = Path(r"E:\GitHub\Confucius4-TTS")
PROMPT_WAV = Path("ref.wav").resolve()  # 在切换 CWD 前固定外部工程中的路径。
OUTPUT_WAV = Path("out.wav").resolve()

os.environ["HF_HUB_CACHE"] = str(REPO_ROOT / "models" / "hf")  # 模型随仓库存放。
os.environ["HF_HUB_OFFLINE"] = "1"  # 模型已缓存时跳过联网。
os.chdir(REPO_ROOT)                 # 当前版本从 CWD 读取 ./checkpoints。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # pip install -e . 后可省略。

import torch
import torchaudio

from confuciustts.cli.inference import ConfuciusTTS

model = ConfuciusTTS(
    config_path=str(REPO_ROOT / "config" / "inference_config.yaml"),
    device="cuda" if torch.cuda.is_available() else "cpu",
)  # 首次加载约 1 分钟；之后 generate() 直接复用。

audio = model.generate(
    text="要合成的文本",
    lang="zh",  # 官方列出的 14 种语言；可与参考音频不同语言。
    prompt_wav=str(PROMPT_WAV),
)
torchaudio.save(
    str(OUTPUT_WAV),
    audio.cpu(),
    model.sample_rate,
    encoding="PCM_S",
    bits_per_sample=16,
)  # 显式写出 22050 Hz / 单声道 / PCM16；本机已验证这两个 save 参数。
```

> `os.chdir()` 是进程级全局状态，多线程 / 多组件工程应谨慎使用。若不能接受切换 CWD，需要先修改 `inference.py`，将配置中的本地资源路径统一重定位到仓库目录。

### `generate()` 全部参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `text` / `lang` / `prompt_wav` | 必填 | 见上 |
| `raw` | `False` | `True` 则跳过文本归一化（自己保证注音 / 数字格式） |
| `temperature` | `0.8` | 表现力：调高（0.9–1.0）更放开但偶有翻车；调低更稳 |
| `top_p` / `top_k` | `0.8 / 30` | 采样窗口 |
| `num_beams` | `3` | 束宽 |
| `repetition_penalty` | `10.0` | 重复惩罚 |
| `max_length` | `1520` | **总序列长度上限**，包含条件、文本、BOS 前缀和生成 token；不是独立的语义 token 配额 |
| `n_timesteps` | `25` | S2A 扩散步数，调高音质略升、变慢 |
| `inference_cfg_rate` | `0.7` | 参考条件引导强度；本机实验中 **1.1** 可加强情绪迁移 |
| `max_text_tokens_per_segment` | `80` | 长文本自动分段阈值 |
| `cross_fade_duration` | `0.3` | 段间处理窗口：约 1/3 淡出、1/3 静音、1/3 淡入；默认即约 0.1s + 0.1s + 0.1s |
| `edge_fade_duration` / `edge_pad_duration` | `0.1 / 0.1` | **当前版本未实际使用（no-op）**，调整不会改变输出 |
| `verbose` | `False` | 输出归一化和分段等日志 |

### 情绪控制要点（本机实验结论）

- **情绪随参考音频克隆**：想要什么情绪就喂什么情绪的参考。可先对参考库做情绪标注（emotion2vec），建索引后按情绪挑选。
- 情绪偏淡时 `inference_cfg_rate=1.1` 补强；>1.3 可能放大参考里的瑕疵（齿音、气声）。
- 兴奋类参考能量大，输出峰值可能超过可编码范围。若需限幅，应在生成 tensor 后、PCM 编码或播放前做峰值归一化，例如缩放到 0.95；已经削波的 PCM 无法靠事后归一化恢复。

### 高级：三件套重组（韵律移植 / 音色混合）

`generate()` 内部把参考音频拆成 **语义特征（韵律/情绪）+ 音色向量 + 参考梅尔** 三件套传给 `_synth_segment`，可以手动混搭“创造”声音：

```python
def parts(model, wav_path):
    """把一段参考音频拆成 (语义特征, 音色向量, 参考梅尔) 三件套。"""
    w16, wt = model._load_prompt(wav_path)          # 16k 波形 + 目标采样率波形
    return (model._extract_semantic(w16),           # 韵律/情绪条件 → T2S
            model._extract_style(w16),              # 音色向量 → S2A
            model._ref_mel(wt))                     # 参考梅尔 → S2A


semA, embA, melA = parts(model, "voiceA.wav")
semB, embB, melB = parts(model, "voiceB.wav")
text = model.normalizer.normalize("要合成的文本", language="zh")
ARGS = (0.8, 0.8, 30, 3, 10.0, 1520, 25, 0.7, False)  # temperature..cfg_rate, verbose

# 韵律移植：B 的说话方式/情绪 + A 的音色。
audio_prosody = model._synth_segment(text, "zh", semB, embA, melA, *ARGS)

# 音色混合：α·A + β·B 的音色向量（α+β=1），说话方式用 A 的。
audio_hybrid = model._synth_segment(text, "zh", semA, 0.5 * embA + 0.5 * embB, melA, *ARGS)
```

> 以上是内部方法（`_` 前缀），随版本可能变化；输出是 `(B, T)` 张量，保存前使用 `.cpu()` 并确保二维。高级示例的 `ARGS` 中 `max_length=1520` 同样遵循上一节的总序列长度语义。

语速：基础后端目前硬编码在 `inference.py` 的 `target_lengths = int(T * 1.72)`，改成 `1.72 / speed` 即得 speed 参数（1.2 = 加速 20%）。这是内部实现实验入口，不是稳定 API；vLLM 代码中也有相应系数，需要分别调整。

---

## 三、CLI 子进程调用

从仓库根目录执行：

```bash
HF_HUB_CACHE="$PWD/models/hf" HF_HUB_OFFLINE=1 python example.py \
    --prompt_wav ref.wav --text "文本" --lang zh --out out.wav \
    --config config/inference_config.yaml
```

退出码 0 = 成功（输出以 `Saved <path>` 开头的日志）；**每次调用重新加载模型（约 1 分钟）**，仅适合偶发调用或 CI 冒烟测试。

注意：`example.py` 直接对 float tensor 调用 `torchaudio.save()`，没有显式指定编码。在本机 soundfile 后端下，产物是 **32-bit float WAV**，不是 PCM16。下游若要求 PCM16，请使用上一节 Python 示例中的显式保存参数，或在 CLI 后增加转码步骤。

---

## 四、参考音频规范

| 项 | 要求 |
|---|---|
| 格式 | 推荐 WAV / FLAC；本机已验证 MP3 / OGG。**本机 torchaudio 只有 soundfile 后端，当前 libsndfile 不支持 M4A**；vLLM 也直接用 soundfile 解码，因此 HTTP 白名单虽含 `.m4a`，实际仍会失败，请先转码 |
| 采样率 | 任意有效输入采样率（内部自动重采样；16k+ 效果好） |
| 声道 | 单/立体声均可（自动混单） |
| 时长 | 建议 **5–30 秒**；vLLM 后端硬截断 30 秒 |
| 内容 | 干净人声、无 BGM/混响；情绪和语速会被一并克隆 |
| 文本 | **不需要 transcript**（transcript-free 架构） |

其他环境的进程内 API 能否读取额外格式取决于 torchaudio 实际后端；不要把“扩展名通过 HTTP 校验”等同于“服务端一定能解码”。

---

## 五、语言与文本注意事项

- 跨语种克隆（中文参考 → 日/英输出）的本机样例已验证可用，音色保持和口音表现良好；这不构成对任意参考素材与语言组合的质量保证。
- **日语专有名词**：归一化按通用词典注音，可能跑偏（如“停雲”→テイクモ）。对发音有要求时**直接写片假名**（ティンユン），或 `raw=True` 跳过归一化。
- 标点是语调的主要来源：`！？…、` 显著影响起伏，可借“！”抬升语气而不改文本内容。
- 长文本自动按 80 token 分段，段间执行淡出、插入静音和淡入；段间语气独立，重要长文建议自行断句多次调用。

---

## 六、资源与部署清单

- **显存**：约 8GB 是本机 **transformers 基础后端、fp32 加载**的参考值，不能直接套用到 vLLM HTTP 服务。vLLM 还保留用于 latent 重算的原生 T2S 模型，并受 `gpu_memory_utilization` 配置影响，应单独实测峰值。
- **CPU**：基础 Python API 可设置 `device="cpu"`，CLI 在无 CUDA 时也会选择 CPU，但速度会明显下降。不要把这一说明当作 vLLM HTTP 服务已经验证的 CPU 部署方案。
- **磁盘 / 模型存放（本机布局）**：模型共约 5.5GB，**随仓库放在 `<仓库根>/models/hf/`**（标准 HF 缓存目录结构，`models/` 已加入 `.gitignore`）。运行时通过 `HF_HUB_CACHE=<仓库根>/models/hf` 指向该目录，不设此变量时会回落到 `~/.cache/huggingface/hub` 并触发重新下载。tokenizer 与 `wav2vec2bert_stats.pt` 还来自仓库的 `checkpoints/`，不能只准备模型目录而省略仓库文件。
- **离线运行**：模型目录和仓库 `checkpoints/` 齐全后设 `HF_HUB_OFFLINE=1`（配合 `HF_HUB_CACHE`），运行过程不联网。
- **首次下载**（新机器）：可设置 `HF_ENDPOINT=https://hf-mirror.com` 后启动自动下载。国内镜像高并发分片下载曾出现损坏或限流，必要时应降低并发并校验文件。
- **ModelScope 替代下载**：可手工获取 `netease-youdao/Confucius4-TTS`、`AI-ModelScope/w2v-bert-2.0`，但当前加载代码使用 Hugging Face API 和缓存布局，**ModelScope 缓存不会被自动识别**。需要自行转换为对应缓存 / 本地路径或调整加载代码，并校验 sha256；其余模型也要另行准备，不能只下载这两个仓库。
- **采样率与声道**：各路径均为 22050 Hz / 单声道。
- **编码格式**：HTTP 非流式为 PCM16 WAV；HTTP 流式为裸 int16-LE；本文 Python 示例显式保存 PCM16；`example.py` 在本机默认保存 32-bit float WAV。接入方不要对所有产物统一套用 PCM16 解析。

---

## 七、常见问题

| 症状 | 原因与处理 |
|---|---|
| 首次启动卡在下载 | 未设镜像/代理；模型与缓存要求见资源清单 |
| 外部 Python 工程初始化时报 `checkpoints` 文件不存在 | 当前版本按 CWD 解析本地资源；从仓库根目录启动，或先修改路径解析代码 |
| `no kernel image is available` | 检查 PyTorch CUDA 构建；本机 5090 + torch 2.7 使用 cu128，不要混装 cu126 |
| M4A 上传后返回 500 | 扩展名白名单不代表可解码；本机请先转成 WAV / FLAC |
| 把 CLI 的 WAV 当 PCM16 读取失败 | 本机 `example.py` 默认生成 float WAV；检查实际 subtype 或显式转码 |
| 输出情绪比参考淡 | 可尝试 `inference_cfg_rate=1.1` 补强 |
| 输出有轻微破音 | 检查峰值，在 PCM 编码或播放前归一化；已削波的数据无法靠缩放修复 |
| 日语专名读音怪 | 归一化猜音；写片假名或 `raw=True` |
| HTTP 上传报 500（Windows） | 检查当前工作目录所在盘根目录下的 `\tmp`，见第一节 |
| 流式播放报错或把异常播成噪声 | 检查 `[ERROR]` 标记、保留跨块残字节，不要直接对每个 HTTP 分块调用 `np.frombuffer` |
| 并发请求很慢或健康检查超时 | 服务没有完整的应用层排队机制，且同步 GPU 阶段会阻塞事件循环；调用方限并发并结合日志判断 |
