from fastapi import FastAPI, WebSocket
import asyncio
import numpy as np
import base64
from pydub import AudioSegment
import io
import uvicorn
from fastapi.responses import HTMLResponse

app = FastAPI()

# Function to decode audio from base64
def decode_audio(data: str) -> np.ndarray:
    # Decode chunk encoded in base64
    byte_samples = base64.b64decode(data.encode("utf-8"))
    # Recover array from bytes
    samples = np.frombuffer(byte_samples, dtype=np.float32)
    return samples

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    audio_buffer = io.BytesIO()  # In-memory buffer to accumulate audio data
    try:
        while True:
            # Receive the base64 encoded audio data
            base64_audio_data = await websocket.receive_text()
            # Decode the audio
            audio_data = decode_audio(base64_audio_data)
            # Write the decoded audio data into the buffer
            audio_buffer.write(audio_data.tobytes())
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        audio_buffer.seek(0)  # Go back to the beginning of the buffer
        print("Converting to WAV...")

        try:
            # Convert the in-memory audio data to WAV format
            audio = AudioSegment.from_file(audio_buffer, format="raw", frame_rate=44100, channels=1, sample_width=2)
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
