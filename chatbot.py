import os
import asyncio
import ollama
import edge_tts
import pygame
import speech_recognition as sr
from faster_whisper import WhisperModel

# --- 設定區 ---
WHISPER_MODEL_SIZE = "small" # 3060 Ti 跑 small 極快
DEVICE = "cuda"              # 強制使用 NVIDIA 顯卡
LLM_MODEL = "llama3"         # 需先執行 ollama run llama3

# 初始化 Whisper (GPU)
print(f"正在加載 Whisper {WHISPER_MODEL_SIZE} 模型於 {DEVICE}...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type="float16")

# 初始化播放器
pygame.mixer.init()

async def speak(text):
    """文字轉語音並播放"""
    print(f"AI 回覆: {text}")
    output_file = "reply.mp3"
    # 這裡雖然用了 edge-tts (需連網)，但它是目前最自然且免費的方案。
    # 若要全離線，可更換為 Piper。
    communicate = edge_tts.Communicate(text, "zh-TW-HsiaoChenNeural")
    await communicate.save(output_file)
    
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)
    pygame.mixer.music.unload()

def listen_and_save():
    """監聽麥克風並將語音存成臨時檔"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1) # 適應環境噪音
        print("\n>>> 正在聽... (請開始說話)")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=10)
            with open("input.wav", "wb") as f:
                f.write(audio.get_wav_data())
            return True
        except Exception as e:
            print(f"未偵測到聲音或發生錯誤: {e}")
            return False

async def main():
    print(f"--- 離線語音機器人啟動 (大腦: {LLM_MODEL}) ---")
    
    while True:
        if listen_and_save():
            # 1. 耳朵: 語音轉文字 (Whisper GPU)
            segments, _ = whisper_model.transcribe("input.wav", beam_size=5)
            user_text = "".join([s.text for s in segments])
            
            if not user_text.strip():
                continue
                
            print(f"你說: {user_text}")

            # 2. 大腦: 本地 LLM (Ollama)
            response = ollama.chat(model=LLM_MODEL, messages=[
                {'role': 'user', 'content': user_text},
            ])
            ai_reply = response['message']['content']

            # 3. 嘴巴: 播放語音
            await speak(ai_reply)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程式已關閉")