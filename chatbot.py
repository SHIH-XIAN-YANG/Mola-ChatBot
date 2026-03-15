import collections
import queue
import threading
import wave
import asyncio
import webrtcvad
import pyaudio
import ollama
import edge_tts
import pygame
from faster_whisper import WhisperModel

# --- 參數設定 ---
VAD_AGGRESSIVENESS = 3  # 1-3，數字越大對噪音越不敏感
SAMPLE_RATE = 16000
FRAME_MS = 30           # 每次處理 30ms 的音訊
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000) * 2 # 16-bit 佔 2 bytes
SILENCE_LIMIT = 30      # 連續 30 個靜音幀 (約 0.9 秒) 則判定說完話

# 初始化模型
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
pygame.mixer.init()

class VoiceAssistant:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                                   input=True, frames_per_buffer=int(SAMPLE_RATE * FRAME_MS / 1000))
        self.history = [{'role': 'system', 'content': '你是一個幽默的台灣室友 Mola，說話簡短且親切。'}]

    def listen(self):
        print("\n👂 正在聽...")
        audio_frames = []
        ring_buffer = collections.deque(maxlen=10) # 緩衝區，捕捉說話前的瞬間
        triggered = False
        silent_chunks = 0

        while True:
            frame = self.stream.read(int(SAMPLE_RATE * FRAME_MS / 1000))
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

                # 判定說完話
                if silent_chunks > SILENCE_LIMIT:
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

    async def speak(self, text):
        print(f"🤖 Mola: {text}")
        path = "reply.mp3"
        comm = edge_tts.Communicate(text, "zh-TW-HsiaoChenNeural")
        await comm.save(path)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
        pygame.mixer.music.unload()

    async def run(self):
        while True:
            audio_path = self.listen()
            
            # ASR
            segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
            user_text = "".join([s.text for s in segments])
            if not user_text.strip(): continue
            print(f"👤 你說: {user_text}")

            # LLM (Memory)
            self.history.append({'role': 'user', 'content': user_text})
            resp = ollama.chat(model='llama3', messages=self.history)
            ai_reply = resp['message']['content']
            self.history.append({'role': 'assistant', 'content': ai_reply})

            # TTS
            await self.speak(ai_reply)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\n掰掰！")