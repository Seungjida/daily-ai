import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np
from pydub import AudioSegment
import io
import uuid
import os


print(torch.cuda.is_available())

music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
#music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cuda" if torch.cuda.is_available() else "cpu")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")


def generate_music(prompt: str, tokens: int) -> str:
    inputs = music_processor(text=[prompt], padding=True, return_tensors="pt")
    
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


    return s3_url
