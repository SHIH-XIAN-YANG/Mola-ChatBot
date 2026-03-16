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
from faster_whisper import WhisperModel
from openwakeword.model import Model

# --- 參數設定 ---
VAD_AGGRESSIVENESS = 3
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
SILENCE_LIMIT = 30

whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
pygame.mixer.init()

class VoiceAssistant:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                                   input=True, frames_per_buffer=int(SAMPLE_RATE * FRAME_MS / 1000))
        
        # 初始化輕量級喚醒詞模型
        # 注意：這裡先載入預設模型作為範例，後續需要替換為自定義的模型
        self.oww_model = Model(wakeword_models=["hey_mola"]) 
        
        self.history = [
            {
                'role': 'system', 
                'content': '你是一個幽默的台灣室友 Mola。請用繁體中文回覆，說話要口語化。重要規則：嚴禁在回覆中使用任何表情符號或特殊符號。'
            }
        ]

    def _listen_blocking(self):
        """將原本的 listen 抽離為單純的同步阻塞函式，供背景執行緒呼叫"""
        print("\n正在聽...")
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
                    print("偵測到人聲...")
                    triggered = True
                    audio_frames.extend(ring_buffer)
            else:
                audio_frames.append(frame)
                if not is_speech:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks > SILENCE_LIMIT:
                    print("說話結束，處理中...")
                    break
        
        path = "input.wav"
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(audio_frames))
        return path

    def _wait_for_wake_word_blocking(self):
        """使用 openwakeword 進行極輕量的背景監聽"""
        print("\n待機中，喊喚醒詞叫醒我...")
        
        # 確保清空之前的緩衝
        self.stream.read(self.stream.get_read_available(), exception_on_overflow=False)
        
        while True:
            # openwakeword 建議每次輸入較大的 chunk (例如 1280 samples)
            pcm = self.stream.read(1280, exception_on_overflow=False)
            audio_data = np.frombuffer(pcm, dtype=np.int16)
            
            # 預測是否出現喚醒詞
            prediction = self.oww_model.predict(audio_data)
            
            # 檢查是否有任何喚醒詞的分數超過門檻 (通常設定在 0.5 左右)
            for mdl_name, score in prediction.items():
                if score > 0.5:
                    print(f"喚醒成功！ (模型: {mdl_name}, 分數: {score})")
                    return True

    async def speak(self, text):
        clean_text = re.sub(r'[^\x00-\x7F\u4e00-\u9fa5\u3000-\u303F\uFF00-\uFFEF]', '', text)
        print(f"Mola: {clean_text}")
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
            # 1. 待機等喚醒詞 (不阻塞主迴圈)
            await asyncio.to_thread(self._wait_for_wake_word_blocking)
            
            # 2. 喚醒後發出提示音或直接回應
            await self.speak("我在")
            
            # 3. 進入傾聽模式收集問題 (不阻塞主迴圈)
            audio_path = await asyncio.to_thread(self._listen_blocking)
            
            # 4. Whisper 語音辨識 (不阻塞主迴圈)
            segments, _ = await asyncio.to_thread(whisper_model.transcribe, audio_path, beam_size=5)
            user_text = "".join([s.text for s in segments]).strip()
            
            if not user_text: 
                continue
                
            print(f"你說: {user_text}")

            # 5. LLM 處理 (不阻塞主迴圈)
            self.history.append({'role': 'user', 'content': user_text})
            
            if len(self.history) > 7:
                self.history = [self.history[0]] + self.history[-6:]
                
            # 將 Ollama 呼叫放入背景執行緒
            resp = await asyncio.to_thread(ollama.chat, model='llama3', messages=self.history)
            ai_reply = resp['message']['content']
            self.history.append({'role': 'assistant', 'content': ai_reply})

            # 6. TTS 播放
            await self.speak(ai_reply)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\n程式已關閉，掰掰！")