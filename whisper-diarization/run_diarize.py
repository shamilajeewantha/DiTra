import subprocess

def run_diarization(audio_file='received_audio.wav'):
    try:
        # Call the diarize.py script with the audio file argument
        result = subprocess.run(
            ['python', 'diarize.py', '-a', audio_file],
            capture_output=True,
            text=True,
            check=True
        )
        print("Diarization output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during diarization:")
        print(e.stderr)

if __name__ == "__main__":
    run_diarization()
