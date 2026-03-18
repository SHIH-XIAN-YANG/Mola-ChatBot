import collections
import wave
import asyncio
import webrtcvad
import pyaudio
import ollama
import edge_tts
import pygame
import re
import numpy as np
import traceback
from faster_whisper import WhisperModel
from openwakeword.model import Model

# --- 參數設定 ---
VAD_AGGRESSIVENESS = 3
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
SILENCE_LIMIT = 30
WAKE_WORD_THRESHOLD = 0.5  # 喚醒詞靈敏度門檻 (0.0 ~ 1.0)

# 初始化模型
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
pygame.mixer.init()

class VoiceAssistant:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                                   input=True, frames_per_buffer=int(SAMPLE_RATE * FRAME_MS / 1000))
        
        self.oww_model = Model(wakeword_models=["hey_mola.onnx"]) 
        
        self.history = [
            {
                'role': 'system', 
                'content': '你是一個傲嬌學生妹 Mola。請用繁體中文回覆，說話要口語化且簡短（盡量在兩三句話內結束）。重要規則：嚴禁在回覆中使用任何表情符號或特殊符號。'
            }
        ]

    def cleanup(self):
        """釋放音訊硬體資源，避免下次執行時 device busy"""
        print("\n🧹 正在釋放硬體資源...")
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def _listen_blocking(self):
        print("\n👂 正在聽...")
        
        # 💡 改良點 1：傾聽前，強制清空底層麥克風的積壓緩衝，避免錄到 TTS 的殘音
        self.stream.read(self.stream.get_read_available(), exception_on_overflow=False)
        
        audio_frames = []
        ring_buffer = collections.deque(maxlen=10)
        triggered = False
        silent_chunks = 0

        while True:
            frame = self.stream.read(int(SAMPLE_RATE * FRAME_MS / 1000), exception_on_overflow=False)
            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

            if not triggered:
                ring_buffer.append(frame)
                if is_speech:
                    print("🎤 偵測到人聲...")
                    triggered = True
                    audio_frames.extend(ring_buffer)
            else:
                audio_frames.append(frame)
                if not is_speech:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks > SILENCE_LIMIT:
                    print("✅ 說話結束，處理中...")
                    break
        
        path = "input.wav"
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(audio_frames))
        return path

    def _wait_for_wake_word_blocking(self):
        print("\n💤 待機中，喊喚醒詞叫醒我...")
        self.stream.read(self.stream.get_read_available(), exception_on_overflow=False)
        
        while True:
            pcm = self.stream.read(1280, exception_on_overflow=False)
            audio_data = np.frombuffer(pcm, dtype=np.int16)
            prediction = self.oww_model.predict(audio_data)
            
            for mdl_name, score in prediction.items():
                if score > WAKE_WORD_THRESHOLD:
                    print(f"🔔 喚醒成功！ (分數: {score:.3f})")
                    return True
    
    def _check_interrupt(self):
        """在播放語音時，快速檢查是否有喚醒詞特徵"""
        available = self.stream.get_read_available()
        if available >= 1280:
            # 🚨 關鍵修正：一次讀出所有可用的聲音，避免堆積延遲
            pcm = self.stream.read(available, exception_on_overflow=False)
            
            # 如果累積的資料太多，只取最後面的 1280 samples (即最新的聲音)
            if len(pcm) > 1280 * 2: # 乘以 2 是因為 16-bit 佔兩個 bytes
                pcm = pcm[-(1280 * 2):]

            audio_data = np.frombuffer(pcm, dtype=np.int16)
            prediction = self.oww_model.predict(audio_data)
            
            for mdl_name, score in prediction.items():
                if score > WAKE_WORD_THRESHOLD + 0.15: 
                    return True
        return False

    async def speak(self, text):
        clean_text = re.sub(r'[^\x00-\x7F\u4e00-\u9fa5\u3000-\u303F\uFF00-\uFFEF]', '', text)
        print(f"🤖 Mola: {clean_text}")
        path = "reply.mp3"
        comm = edge_tts.Communicate(clean_text, "zh-TW-HsiaoChenNeural")
        await comm.save(path)
        
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        
        # 💡 優化：稍微等待喇叭開始出聲後，再清空麥克風
        await asyncio.sleep(0.1) 
        self.stream.read(self.stream.get_read_available(), exception_on_overflow=False)
        
        interrupted = False
        while pygame.mixer.music.get_busy():
            # 背景檢查是否有人喊 Mola 打斷
            if await asyncio.to_thread(self._check_interrupt):
                print("\n🛑 檢測到打斷指令，Mola 停止說話！")
                pygame.mixer.music.stop()
                interrupted = True
                break
            await asyncio.sleep(0.05)
            
        pygame.mixer.music.unload()
        return interrupted

    async def run(self):
        interrupted = False 
        
        while True:
            # 1. 待機等喚醒詞 (如果剛剛是被打斷的，代表使用者已經喊過 Mola 了，跳過待機)
            if not interrupted:
                await asyncio.to_thread(self._wait_for_wake_word_blocking)
                await self.speak("我在")
            
            interrupted = False 
            
            # 2. 進入傾聽模式收集問題
            audio_path = await asyncio.to_thread(self._listen_blocking)
            
            # 3. Whisper 語音辨識
            try:
                segments, _ = await asyncio.to_thread(whisper_model.transcribe, audio_path, beam_size=5)
                user_text = "".join([s.text for s in segments]).strip()
                
                if not user_text: 
                    continue
                    
                print(f"👤 你說: {user_text}")

                # 4. LLM 處理
                self.history.append({'role': 'user', 'content': user_text})
                if len(self.history) > 7:
                    self.history = [self.history[0]] + self.history[-6:]
                    
                resp = await asyncio.to_thread(ollama.chat, model='llama3', messages=self.history)
                ai_reply = resp['message']['content']
                self.history.append({'role': 'assistant', 'content': ai_reply})

                # 5. TTS 播放 (並接收是否被打斷的狀態)
                interrupted = await self.speak(ai_reply)
                
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")
                traceback.print_exc()
                await self.speak("抱歉，我剛剛腦袋卡住了。")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\n收到中斷指令，程式準備關閉...")
    finally:
        # 💡 改良點 2：確保無論如何都會釋放硬體資源
        assistant.cleanup()
        print("掰掰！")