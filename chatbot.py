import collections
import threading
import wave
import asyncio
import webrtcvad
import pyaudio
import ollama
import edge_tts
import pygame
import re  # 新增：用於正則表達式過濾表情符號
from faster_whisper import WhisperModel

# --- 參數設定 ---
VAD_AGGRESSIVENESS = 3  # 1-3，數字越大對噪音越不敏感
SAMPLE_RATE = 16000
FRAME_MS = 30           # 每次處理 30ms 的音訊
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000) * 2 # 16-bit 佔 2 bytes
SILENCE_LIMIT = 30      # 連續 30 個靜音幀 (約 0.9 秒) 則判定說完話

# 初始化模型 (3060 Ti 使用 cuda)
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
pygame.mixer.init()

class VoiceAssistant:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                                   input=True, frames_per_buffer=int(SAMPLE_RATE * FRAME_MS / 1000))
        self.history = [
            {
                'role': 'system', 
                'content': '你是一個幽默的台灣室友 Mola。請用繁體中文回覆，說話要口語化。重要規則：嚴禁在回覆中使用任何表情符號或特殊符號。'
            }
        ]

    def listen(self, quiet=False):
        """
        quiet=True: 用於待機喚醒模式，不印出提示詞
        quiet=False: 用於正式對話模式
        """
        if not quiet:
            print("\n👂 正在聽...")
            
        audio_frames = []
        ring_buffer = collections.deque(maxlen=10) # 緩衝區，捕捉說話前的瞬間
        triggered = False
        silent_chunks = 0

        while True:
            # 加上 exception_on_overflow=False 防止長時間待機時 PyAudio 報錯崩潰
            frame = self.stream.read(int(SAMPLE_RATE * FRAME_MS / 1000), exception_on_overflow=False)
            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

            if not triggered:
                ring_buffer.append(frame)
                if is_speech:
                    if not quiet:
                        print("🎤 偵測到人聲...")
                    triggered = True
                    audio_frames.extend(ring_buffer)
            else:
                audio_frames.append(frame)
                if not is_speech:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                # 判定說完話
                if silent_chunks > SILENCE_LIMIT:
                    if not quiet:
                        print("✅ 說話結束，處理中...")
                    break
        
        # 儲存暫存檔供 Whisper 讀取
        path = "input.wav"
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(audio_frames))
        return path

    async def wait_for_wake_word(self):
        print("\n💤 待機中，喊『Mola』喚醒我...")
        while True:
            # 使用靜音模式錄音，避免終端機洗版
            audio_path = self.listen(quiet=True) 
            segments, _ = whisper_model.transcribe(audio_path, beam_size=2)
            text = "".join([s.text for s in segments]).strip()
            text_lower = text.lower()
            
            # 加入多種拼音與中文翻譯容錯
            if "mola" in text_lower or "莫拉" in text_lower or "摩拉" in text_lower:
                print(f"🔔 喚醒成功！ (聽到: {text})")
                
                # 判斷使用者是否「一口氣說完」(例如: Mola今天天氣如何)
                # 利用正則表達式把喚醒詞移除，看看剩下什麼
                content = re.sub(r'(?i)(mola|莫拉|摩拉)[,，。 \n]*', '', text).strip()
                
                if len(content) >= 2: 
                    # 如果後面跟著具體問題，直接回傳問題內容
                    return content
                else:
                    # 如果只有喊名子，就出聲回應並等待下一步
                    await self.speak("我在，請問有什麼事嗎？")
                    return None 

    async def speak(self, text):
        # 雙重保險：強制過濾所有非標準文字的符號（徹底消滅表情符號）
        clean_text = re.sub(r'[^\x00-\x7F\u4e00-\u9fa5\u3000-\u303F\uFF00-\uFFEF]', '', text)
        
        print(f"🤖 Mola: {clean_text}")
        path = "reply.mp3"
        comm = edge_tts.Communicate(clean_text, "zh-TW-HsiaoChenNeural")
        await comm.save(path)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
        pygame.mixer.music.unload()

    async def run(self):
        while True:
            # 1. 待機等喚醒詞
            user_text = await self.wait_for_wake_word()
            
            # 2. 如果只是單純喚醒，就進入傾聽模式收集問題
            if user_text is None:
                audio_path = self.listen(quiet=False)
                segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
                user_text = "".join([s.text for s in segments]).strip()
            
            if not user_text: 
                continue # 如果沒聽到聲音，退回待機模式
                
            print(f"👤 你說: {user_text}")

            # 3. LLM (Memory)
            self.history.append({'role': 'user', 'content': user_text})
            
            # 記憶體管理：避免對話太長導致 VRAM 爆炸 (保留 System prompt + 最近 6 次對話)
            if len(self.history) > 7:
                self.history = [self.history[0]] + self.history[-6:]
                
            resp = ollama.chat(model='llama3', messages=self.history)
            ai_reply = resp['message']['content']
            self.history.append({'role': 'assistant', 'content': ai_reply})

            # 4. TTS 播放 (播完後會自動進入下一次 while 迴圈的待機模式)
            await self.speak(ai_reply)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\n程式已關閉，掰掰！")