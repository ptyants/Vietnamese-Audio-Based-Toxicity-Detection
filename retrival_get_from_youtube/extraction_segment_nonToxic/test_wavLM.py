import torch
import librosa
from transformers import AutoFeatureExtractor, WavLMModel

# Load model
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "microsoft/wavlm-base-plus"
)

model = WavLMModel.from_pretrained(
    "microsoft/wavlm-base-plus"
)

# Đọc audio
audio, sr = librosa.load(
    r".\Merch_Datasets\wav_store\vitosa_train_2951.wav",
    sr=16000
)

# Chuẩn bị input
inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

# Trích xuất embedding
with torch.no_grad():
    outputs = model(**inputs)

# (1, sequence_length, 768)
hidden_states = outputs.last_hidden_state

print(hidden_states.shape)