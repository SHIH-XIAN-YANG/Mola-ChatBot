# Mola ChatBot

一個以語音互動為核心的本地 AI 助理：

- 使用 **喚醒詞（Mola／莫拉／摩拉）** 進入對話
- 透過 **faster-whisper** 進行語音轉文字（STT）
- 透過 **Ollama + llama3** 產生回覆
- 透過 **edge-tts** 進行文字轉語音（TTS）
- 預設角色是「幽默的台灣室友 Mola」，並以繁體中文口語回覆

## 功能特色

- **待機喚醒模式**：持續聆聽，聽到喚醒詞才進入互動
- **一句話喚醒 + 問句容錯**：例如「Mola 今天天氣如何」可直接當作問題處理
- **VAD 端點偵測**：用 `webrtcvad` 判斷說話開始/結束，減少空白錄音
- **對話記憶管理**：保留 System Prompt 與最近幾輪對話，避免上下文過長
- **回覆文字清理**：會先過濾非預期符號再 TTS 播放

## 專案結構

```text
.
├── chatbot.py        # 主程式（錄音、喚醒詞、STT、LLM、TTS）
├── requirements.txt  # Python 相依套件
└── LICENSE
```

## 系統需求

建議環境：

- Python 3.10+
- 可用麥克風與音訊輸出裝置
- 已安裝並可執行 [Ollama](https://ollama.com/)
- 已下載模型（預設 `llama3`）
- NVIDIA GPU（可選，但目前程式預設用 CUDA 跑 Whisper）

> 注意：`chatbot.py` 目前以 `WhisperModel("small", device="cuda", compute_type="float16")` 初始化，若你沒有 CUDA GPU，需改成 CPU 設定。

## 安裝

1. 下載專案

```bash
git clone <your-repo-url>
cd Mola-ChatBot
```

2. 建立虛擬環境並安裝依賴

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

3. 安裝並啟動 Ollama，拉取模型

```bash
ollama pull llama3
```

## 使用方式

啟動程式：

```bash
python chatbot.py
```

執行後流程：

1. 程式進入待機模式（顯示「喊 Mola 喚醒我」）
2. 你說出「Mola／莫拉／摩拉」喚醒
3. 說出問題（或在喚醒詞後直接接問題）
4. 程式將語音轉文字、送至 Ollama 生成回覆並語音播報

## 主要參數（可在 `chatbot.py` 調整）

- `VAD_AGGRESSIVENESS = 3`：VAD 嚴格度（1–3）
- `SAMPLE_RATE = 16000`：取樣率
- `FRAME_MS = 30`：每幀音訊長度（ms）
- `SILENCE_LIMIT = 30`：連續靜音幀數門檻（判定說話結束）

## 常見問題

### 1) 沒有 CUDA GPU，程式一啟動就報錯

把這行：

```python
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
```

改成（較慢但可用）：

```python
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
```

### 2) 聽不到聲音或播放失敗

- 確認系統音訊輸出裝置正常
- 確認 `pygame` 可正常初始化 mixer
- 某些 Linux 環境需額外安裝音訊底層套件（如 ALSA/PulseAudio）

### 3) 無法辨識喚醒詞

- 拉近麥克風、降低背景噪音
- 放慢語速
- 確認輸入語言為中文情境

## 已知限制

- 目前為單檔腳本，尚未模組化
- 語音檔使用固定檔名（`input.wav`、`reply.mp3`）
- 回覆語言與人格設定寫死在 system prompt
- 對話記憶採簡單截斷策略

## 授權

本專案採用 MIT License，詳見 [LICENSE](./LICENSE)。