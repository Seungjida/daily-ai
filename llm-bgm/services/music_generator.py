import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np
from pydub import AudioSegment
import io
import uuid
import os
from app.utils import upload_to_s3

music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small", attn_implementation="eager"
).to("cuda" if torch.cuda.is_available() else "cpu")

def generate_music(prompt: str, tokens: int) -> str:
    inputs = music_processor(text=[prompt], padding=True, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        audio_values = music_model.generate(**inputs, max_new_tokens=tokens)

    sampling_rate = music_model.config.audio_encoder.sampling_rate
    audio_data = audio_values[0, 0].cpu().numpy().astype(np.float32)

    fade_out_duration = 2 
    fade_out_samples = fade_out_duration * sampling_rate 

    fade_out = np.exp(np.linspace(0, -4, fade_out_samples))

    if len(audio_data) >= fade_out_samples: 
        audio_data[-fade_out_samples:] *= fade_out
    else:
        print("오디오 길이가 너무 짧아 페이드 아웃을 적용할 수 없습니다.")

    mp3_path = os.path.join(os.path.dirname(__file__), f"{uuid.uuid4()}.mp3")

    try:
        wav_io = io.BytesIO()
        scipy.io.wavfile.write(wav_io, rate=sampling_rate, data=audio_data)
        wav_io.seek(0)
        audio = AudioSegment.from_wav(wav_io)

        target_dBFS = -12.0
        change_in_dBFS = target_dBFS - audio.dBFS 
        audio = audio.apply_gain(change_in_dBFS)

        audio.export(mp3_path, format="mp3", bitrate="128k")

        s3_object_name = f"{uuid.uuid4()}.mp3"

        s3_url = upload_to_s3(mp3_path, s3_object_name)
    except Exception as e:
        print(f"S3에 업로드 도중 에러가 발생했습니다: {e}")
        raise

    os.remove(mp3_path)

    return s3_url
