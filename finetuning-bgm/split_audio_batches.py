import os
from pydub import AudioSegment

def split_audio_to_chunks(input_path, output_dir, chunk_length_sec=30, discard_short=True):
    """
    - input_path: 원본 오디오 파일 경로
    - output_dir: 분할된 30초 파일 저장 폴더
    - chunk_length_sec: 30초 단위로 분할
    - discard_short: 마지막 구간이 너무 짧으면 버릴지 여부
    """
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_file(input_path)

    # WAV (44100Hz, 스테레오) 변환
    audio = audio.set_frame_rate(44100).set_channels(2)

    chunk_paths = []
    chunk_length_ms = chunk_length_sec * 1000
    total_length_ms = len(audio)

    start = 0
    idx = 0
    while start < total_length_ms:
        end = start + chunk_length_ms
        if end > total_length_ms:
            # leftover가 10초 미만이면 버리기
            if discard_short and (total_length_ms - start) < (chunk_length_sec * 1000 / 3):
                break
            end = total_length_ms

        chunk_audio = audio[start:end]
        chunk_name = f"chunk_{idx:03d}.wav"
        chunk_path = os.path.join(output_dir, chunk_name)
        chunk_audio.export(chunk_path, format="wav")

        chunk_paths.append(chunk_path)
        idx += 1
        start += chunk_length_ms

    return chunk_paths

# 모든 오디오 파일을 순회하면서 자동으로 30초 단위로 분할
audio_folder = "/content/drive/MyDrive/audio_samples"
output_base_folder = "/content/drive/MyDrive/audio_samples"

# 지원하는 오디오 파일 확장자 목록
valid_extensions = (".wav")

# 폴더 내 모든 오디오 파일 찾기
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(valid_extensions)]

if not audio_files:
    print("처리할 오디오 파일이 없습니다.")
else:
    print(f"{len(audio_files)}개의 오디오 파일을 찾았습니다! 자동 처리 시작...")

    for audio_file in audio_files:
        input_path = os.path.join(audio_folder, audio_file)
        output_folder = os.path.join(output_base_folder, f"chunks_{os.path.splitext(audio_file)[0]}")

        print(f"{audio_file} → {output_folder} 에 30초 단위로 분할 중...")
        chunk_files = split_audio_to_chunks(input_path, output_folder, chunk_length_sec=30, discard_short=True)

        print(f"{len(chunk_files)}개의 30초 WAV 파일 생성 완료!")

print("모든 오디오 파일 분할이 완료되었습니다!")
