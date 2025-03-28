from fastapi import FastAPI, WebSocket
import asyncio
import numpy as np
import base64
import io
import uvicorn
from pydub import AudioSegment
from fastapi.responses import HTMLResponse

app = FastAPI()

# Function to decode and convert audio
def decode_audio(data: str) -> np.ndarray:
    byte_samples = base64.b64decode(data.encode("utf-8"))
    samples = np.frombuffer(byte_samples, dtype=np.float32)

    # Convert float32 PCM to int16 PCM
    samples = np.int16(samples * 32767)  # Normalize and convert
    return samples

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    audio_buffer = io.BytesIO()  # In-memory buffer to accumulate audio data
    try:
        while True:           
            base64_audio_data = await websocket.receive_text()
            audio_data = decode_audio(base64_audio_data)
            audio_buffer.write(audio_data.tobytes())  # Store as raw PCM
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        audio_buffer.seek(0)  # Reset buffer position
        print("Converting to WAV...")

        try:
            audio = AudioSegment.from_raw(
                audio_buffer, 
                sample_width=2,  # 16-bit PCM
                frame_rate=44100,
                channels=1
            )
            audio.export("received_audio.wav", format="wav")
            print("Saved as received_audio.wav")
        except Exception as e:
            print(f"Error during conversion: {e}")

# Serve the frontend HTML file
@app.get("/")
async def get():
    return HTMLResponse(content=open("index.html").read(), status_code=200)

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
