import multiprocessing
import os
import signal
import sys
import time
import wave
import whisper
from concurrent.futures import ProcessPoolExecutor, as_completed

chunk_folder = "copied_chunks"

def get_audio_duration(filepath):
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception as e:
        return 0.0

def transcribe_chunk(chunk_path):
    try:
        model = whisper.load_model("tiny")
        print(f"Model loaded in process for {chunk_path}")

        duration = get_audio_duration(chunk_path)
        result = model.transcribe(chunk_path, word_timestamps=True)

        print(f"\nTranscribing: {chunk_path}")
        print(f"Chunk Time: 0.00s - {duration:.2f}s")

        segments_text = []
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            segment_str = f"{start:.2f}s - {end:.2f}s : {text}"
            print(segment_str)
            segments_text.append(segment_str)

        return (chunk_path, segments_text)

    except Exception as e:
        return (chunk_path, f"[ERROR]: {str(e)}")

if __name__ == "__main__":
    # Enable killing of the entire process group on Ctrl+C
    os.setpgrp()  # Make this process the leader of a new process group

    chunk_files = sorted([
        os.path.join(chunk_folder, f)
        for f in os.listdir(chunk_folder)
        if f.endswith(".wav")
    ])

    print(f"Found {len(chunk_files)} chunk files")
    if len(chunk_files) == 0:
        print("No .wav files found in folder. Exiting.")
        exit(1)

    start_time = time.time()

    max_workers = 6
    print(f"Using max_workers = {max_workers}")

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(transcribe_chunk, path) for path in chunk_files]

            for future in as_completed(futures):
                chunk_path, segments = future.result()
                if isinstance(segments, str) and segments.startswith("[ERROR]"):
                    print(f"Error in {chunk_path}: {segments}")
                else:
                    print(f"\nResults for {chunk_path}:")
                    for segment in segments:
                        print(segment)

    except KeyboardInterrupt:
        print("\n🔴 Transcription interrupted by user (Ctrl+C)")
        # Kill entire process group
        os.killpg(0, signal.SIGTERM)  # Sends SIGTERM to all in the process group
        sys.exit(1)

    print(f"\n⏱️ Total transcription time: {time.time() - start_time:.2f} seconds")
