import logging
from fastapi import FastAPI, WebSocket
import numpy as np
import base64
from pydub import AudioSegment
import io
import uvicorn
from fastapi.responses import HTMLResponse
import time
import sys
import torchaudio
import whisper
import json

DURATION = 30  # Duration in seconds for which audio is saved

# Set up the first logger for transcription log
logger1 = logging.getLogger("logger1")
logger1.setLevel(logging.INFO)
file_handler1 = logging.FileHandler("transcription_log.txt")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler1.setFormatter(formatter)
logger1.addHandler(file_handler1)

# Set up the second logger for a second log
logger2 = logging.getLogger("logger2")
logger2.setLevel(logging.INFO)
file_handler2 = logging.FileHandler("diarization_log.txt")
file_handler2.setFormatter(formatter)
logger2.addHandler(file_handler2)

app = FastAPI()

model = whisper.load_model("tiny")

diarization_data = []
diarization_start_time = None
local_start_time = 0
previous_speaker = None
accumulated_duration = 0.0


def whisper_transcribe(waveform, sample_rate):
    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    audio = waveform.squeeze().numpy()    

    result = model.transcribe(audio, word_timestamps=True)

    logger1.info(f"Transcription result with speaker tags:")

    # Loop through each segment in the transcription
    for segment in result['segments']:
        for word_info in segment['words']:
            word_start = float(word_info["start"])
            word_end = float(word_info["end"])
            word = word_info["word"]
            probability = float(word_info["probability"])

            # Default speaker if none is matched
            speaker = "unknown"

            # Match with diarization data
            for entry in diarization_data:
                diar_start = entry["start_time"]
                diar_end = diar_start + entry["duration"]

                # If the word falls within this diarization segment
                if diar_start <= word_start <= diar_end:
                    speaker = entry["speaker_label"]
                    break

            logger1.info(json.dumps({
                "word": word,
                "start": word_start,
                "end": word_end,
                "probability": probability,
                "speaker": speaker
            }))



# Function to decode audio from base64
def decode_audio(data: str) -> np.ndarray:
    # Decode chunk encoded in base64
    byte_samples = base64.b64decode(data.encode("utf-8"))
    # Recover array from bytes
    samples = np.frombuffer(byte_samples, dtype=np.float32)
        
    samples = np.int16(samples * 32767)  # Normalize and convert

    return samples

# Function to save audio when buffer hits 5 seconds
def save_audio_from_buffer(audio_buffer: io.BytesIO, start_time: float, end_time: float):
    audio_buffer.seek(0)  # Go back to the beginning of the buffer
    try:
        # Convert the in-memory audio data to WAV format
        audio = AudioSegment.from_raw(
                audio_buffer, 
                sample_width=2,  # 16-bit PCM
                frame_rate=44100,
                channels=1
            )
        # Export to the in-memory buffer in WAV format
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)  # Go back to the beginning of the buffer

        # Load audio directly from the WAV in-memory buffer using torchaudio
        waveform, sample_rate = torchaudio.load(wav_buffer)
        whisper_transcribe(waveform, sample_rate)

    except Exception as e:
        logger1.error(f"Error during conversion: {e}")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger1.info("Client connected")

    audio_buffer = io.BytesIO()  # In-memory buffer to accumulate audio data
    start_time = 0  # Record the start time of the buffer
    buffer_duration = 0  # Duration of the accumulated audio data

    try:
        while True:
            try:
                # Receive the base64 encoded audio data
                base64_audio_data = await websocket.receive_text()
                if base64_audio_data:
                    # Decode the audio
                    audio_data = decode_audio(base64_audio_data)

                    # Write the decoded audio data into the buffer
                    audio_buffer.write(audio_data.tobytes())

                    # Calculate the duration of the accumulated audio in seconds
                    buffer_duration = len(audio_buffer.getvalue()) / (44100 * 2)  # 44100 samples per second, 2 bytes per sample

                    # If the buffer reaches 5 seconds, save it and reset
                    if buffer_duration >= DURATION:
                        logger1.info(f"{DURATION} seconds reached. Saving audio...")
                        end_time = start_time + DURATION
                        save_audio_from_buffer(audio_buffer, start_time, end_time)
                        audio_buffer.seek(0)  # Reset the buffer
                        audio_buffer.truncate(0)  # Clear the buffer
                        start_time = end_time  # Reset the start time
                        buffer_duration = 0  # Reset the duration
                else:
                    logger1.warning("No audio data received. Check frontend sending.")
            
            except Exception as e:
                logger1.error(f"Error receiving audio: {e}")
                break

    except Exception as e:
        logger1.error(f"Connection closed: {e}")
    finally:
        if buffer_duration > 0:  # Save any remaining audio when the connection ends
            logger1.info("Connection closed. Saving remaining audio...")
            end_time = start_time + buffer_duration
            save_audio_from_buffer(audio_buffer, start_time, end_time)


# WebSocket endpoint
@app.websocket("/ws_diar")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger2.info("Diarization Client connected")

    try:
        while True:
            try:
                # Receive the base64 encoded audio data
                diar_data = await websocket.receive_text()
                if diar_data:
                    lines = diar_data.splitlines()
                    for line in lines:
                        global diarization_data, diarization_start_time, previous_speaker, accumulated_duration, local_start_time
                        # logger2.info(f"Processed diarization data: {line}")
                        parts = line.split() 

                        try:
                            # Extract start time, end time, and speaker label (assuming they are in fixed positions)
                            start_time = float(parts[3])  # Start time is in position 3
                            duration = float(parts[4])    # End time is in position 4
                            speaker_label = parts[7]  # Speaker label is in position 7

                            if diarization_start_time is None:
                                diarization_start_time = start_time
                                local_start_time = start_time
                                logger2.info(f"Start time: {diarization_start_time}")

                            if previous_speaker is None:
                                previous_speaker = speaker_label

                            if speaker_label == previous_speaker:
                                accumulated_duration += duration
                            else:
                                # Append the extracted values to the diarization_data list
                                diarization_data.append({
                                    'start_time': round(local_start_time-diarization_start_time, 2),
                                    'duration': round(accumulated_duration, 2),
                                    'speaker_label': previous_speaker
                                })

                                logger2.info(f"Processed diarization data: Start time: {diarization_data[-1]['start_time']}, Duration : {diarization_data[-1]['duration']}, Speaker: {diarization_data[-1]['speaker_label']}")

                                previous_speaker = speaker_label
                                accumulated_duration = duration
                                local_start_time = start_time

                        except ValueError as e:
                            logger2.error(f"Error parsing line: {line}, Error: {str(e)}")

                        if diarization_start_time is None:
                            diarization_start_time = start_time
                            logger2.info(f"Start time: {diarization_start_time}")                            
            except Exception as e:
                logger2.error(f"Error receiving diarization data: {e}")
                break
    except Exception as e:
        logger2.error(f"Diarization Connection closed: {e}")
    finally:
        logger2.info("Diarization Connection closed.")



# Serve the frontend HTML file
@app.get("/")
async def get():
    return HTMLResponse(content=open("index.html", encoding="utf-8").read(), status_code=200)

# Run the application with log level set to debug
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
