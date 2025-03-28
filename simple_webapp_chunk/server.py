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

DURATION = 10  # Duration in seconds for which audio is saved

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG to capture more details
logger = logging.getLogger(__name__)

app = FastAPI()

# Function to decode audio from base64
def decode_audio(data: str) -> np.ndarray:
    # Decode chunk encoded in base64
    byte_samples = base64.b64decode(data.encode("utf-8"))
    
    # Recover array from bytes
    samples = np.frombuffer(byte_samples, dtype=np.float32)
        
    # Convert float32 PCM to int16 PCM
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
        # Save the audio with the given time range in the filename
        file_name = f"received_audio_{int(start_time)}-{int(end_time)}s.wav"
        audio.export(file_name, format="wav")
        logger.info(f"Saved as {file_name}")
    except Exception as e:
        logger.error(f"Error during conversion: {e}")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

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
                        logger.info(f"{DURATION} seconds reached. Saving audio...")
                        end_time = start_time + DURATION
                        save_audio_from_buffer(audio_buffer, start_time, end_time)
                        audio_buffer.seek(0)  # Reset the buffer
                        audio_buffer.truncate(0)  # Clear the buffer
                        start_time = end_time  # Reset the start time
                        buffer_duration = 0  # Reset the duration
                else:
                    logger.warning("No audio data received. Check frontend sending.")
            
            except Exception as e:
                logger.error(f"Error receiving audio: {e}")
                break

    except Exception as e:
        logger.error(f"Connection closed: {e}")
    finally:
        if buffer_duration > 0:  # Save any remaining audio when the connection ends
            logger.info("Connection closed. Saving remaining audio...")
            end_time = start_time + buffer_duration
            save_audio_from_buffer(audio_buffer, start_time, end_time)

# Serve the frontend HTML file
@app.get("/")
async def get():
    return HTMLResponse(content=open("index.html").read(), status_code=200)

# Run the application with log level set to debug
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
