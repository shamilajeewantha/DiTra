import whisper
import numpy as np
import torchaudio
import json

model = whisper.load_model("tiny")

# result = model.transcribe("audio.wav", word_timestamps=True)

waveform, sample_rate = torchaudio.load("audio.wav")
print(waveform.shape)
print(sample_rate)

# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)


waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
audio = waveform.squeeze().numpy()

# Define chunk size (30 seconds) and Whisper's sample rate (16 kHz)
chunk_size = 30 * 16000  # 30 seconds * 16,000 samples per second
num_chunks = int(np.ceil(len(audio) / chunk_size))

for i in range(num_chunks):
    print(f"Processing chunk {i + 1}/{num_chunks}...")
    start_sample = i * chunk_size
    end_sample = min((i + 1) * chunk_size, len(audio))
    chunk = audio[start_sample:end_sample]
    result = model.transcribe(chunk, word_timestamps=True)

    print(result["text"])

    # Assuming `result` contains the JSON-like dictionary
    for segment in result['segments']:
        for word_info in segment['words']:
            print(json.dumps({
                "word": word_info["word"],
                "start": float(word_info["start"]),
                "end": float(word_info["end"]),
                "probability": float(word_info["probability"])
            }))



# import whisper

# model = whisper.load_model("tiny")

# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("audio.wav")
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)