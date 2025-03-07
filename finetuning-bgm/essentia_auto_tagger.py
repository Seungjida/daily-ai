import os
import json
import numpy as np
import essentia.standard as es
import librosa
import torch
import subprocess as sp

# Colab에서 GPU 사용 설정 (없으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Essentia 모델 다운로드 (이미 존재하면 다운로드 안 함)
MODEL_PATHS = {
    "embedding": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "genre": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "mood": "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "instrument": "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb"
}

for model_name, url in MODEL_PATHS.items():
    model_file = url.split("/")[-1]
    if not os.path.exists(model_file):
        print(f"Downloading {model_name} model")
        sp.call(["wget", "-nc", url])  # 중복 다운로드 방지

# 기본 태그 리스트 (metadata.py가 없을 경우)
try:
    from metadata import genre_labels, mood_theme_classes, instrument_classes
except ImportError:
    print("`metadata.py`가 없어서 기본 태그 리스트를 사용합니다.")
    genre_labels = ["Rock", "Pop", "Jazz", "Classical", "Hip-hop", "Electronic", "Folk", "Blues", "Reggae", "Soul"]
    mood_theme_classes = ["Happy", "Sad", "Energetic", "Calm", "Dark", "Bright", "Romantic", "Aggressive"]
    instrument_classes = ["Piano", "Guitar", "Violin", "Drums", "Synthesizer", "Bass", "Saxophone", "Flute"]

def filter_top_tags_with_threshold(predictions, class_list, threshold=0.1, min_tags=3):
    """Threshold 이상 태그 필터링하되, 최소 min_tags 개수는 유지"""
    predictions_mean = np.mean(predictions, axis=0)
    sorted_indices = np.argsort(predictions_mean)[::-1]
    filtered_indices = [i for i in sorted_indices if predictions_mean[i] > threshold]

    # 최소 태그 개수를 만족시키기 위해 부족하면 추가
    while len(filtered_indices) < min_tags and len(filtered_indices) < len(class_list):
        next_best = sorted_indices[len(filtered_indices)]
        filtered_indices.append(next_best)

    return [class_list[i] for i in filtered_indices]

# 모델들을 한 번만 로드하여 재사용
embedding_model = es.TensorflowPredictEffnetDiscogs(
    graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1"
)
genre_model = es.TensorflowPredict2D(
    graphFilename="genre_discogs400-discogs-effnet-1.pb",
    input="serving_default_model_Placeholder",
    output="PartitionedCall:0"
)
mood_model = es.TensorflowPredict2D(
    graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    input="model/Placeholder",
    output="model/Sigmoid"
)
instrument_model = es.TensorflowPredict2D(
    graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb",
    input="model/Placeholder",
    output="model/Sigmoid"
)

def get_audio_features(audio_filename):
    """오디오 파일에서 특징을 추출하고, 자동 태깅을 수행하는 함수"""
    try:
        # 오디오 파일 로드 (16000Hz, mono)
        audio = es.MonoLoader(filename=audio_filename, sampleRate=16000)()
        embeddings = embedding_model(audio)

        result_dict = {}

        # 장르 예측
        predictions = genre_model(embeddings)
        result_dict['genres'] = filter_top_tags_with_threshold(predictions, genre_labels)

        # 분위기(Mood) 예측
        predictions = mood_model(embeddings)
        result_dict['moods'] = filter_top_tags_with_threshold(predictions, mood_theme_classes)

        # 악기 구성 예측
        predictions = instrument_model(embeddings)
        result_dict['instruments'] = filter_top_tags_with_threshold(predictions, instrument_classes)

        # BPM 및 키(Key) 추출
        y, sr = librosa.load(audio_filename)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # 만약 tempo가 NumPy 배열(numpy.ndarray)이라면 스칼라(float)로 변환
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo)
        tempo = round(tempo)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = np.argmax(np.sum(chroma, axis=1))
        key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key_index]

        result_dict["bpm"] = tempo
        result_dict["key"] = key

        return result_dict

    except Exception as e:
        print(f"오류 발생! {audio_filename} 처리 중 오류: {str(e)}")
        return None

# 처리된 파일 목록 (중복 처리 방지)
processed_files_path = "processed_files.json"
if os.path.exists(processed_files_path):
    with open(processed_files_path, "r") as pf:
        processed_files = set(json.load(pf))
else:
    processed_files = set()

# 결과 저장할 JSONL 파일 경로
base_folder = "/content/drive/MyDrive/audio_samples"
output_json = "/content/drive/MyDrive/audio_samples/essentia_auto_tagging_results.jsonl"

# 모든 chunks_* 폴더 순회하여 분석 진행
chunk_folders = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if f.startswith("chunks_")]

with open(output_json, "a", encoding="utf-8") as f:
    for chunk_folder in chunk_folders:
        if not os.path.isdir(chunk_folder):
            continue

        print(f"Processing folder: '{chunk_folder}'")
        chunk_files = [os.path.join(chunk_folder, file) for file in os.listdir(chunk_folder) if file.endswith(".wav")]

        for wav_file in chunk_files:
            if wav_file in processed_files:
                print(f"Skipping already processed file: {wav_file}")
                continue

            print(f"Analyzing: {wav_file}")
            mood_features = get_audio_features(wav_file)

            if mood_features:
                result = {"audio_path": wav_file, "mood_features": mood_features}
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

                # 처리된 파일 목록 업데이트
                processed_files.add(wav_file)
                with open(processed_files_path, "w") as pf:
                    json.dump(list(processed_files), pf)

print(f"모든 WAV 파일 분석 완료: {output_json}")
