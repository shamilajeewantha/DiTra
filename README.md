# DiTra

## simple web app received audio and records it and save it as a wav file at the end

## chunking webapp saves when the recording hits 30s so for the duration of the recording, we get multiple wav files representing chunks

## whisper webapp sends 30s chunks through the models and logs transcription instead of saving them as wav.

## next I will do VAD

python .\server.py
diart.serve --host 0.0.0.0 --port 7007

Here is a sample `README.md` file that provides clear instructions on how to run the provided Python commands:

---

# 🖥️ Server Setup & Running Instructions

This guide explains how to set up and run the server using `server.py` and `diart.serve`.

## 📦 Requirements

Make sure you have the following installed:

* Python 3.8 or higher

## 🛠️ Installation

1. **Clone the Repository:**

```bash
git clone https://github.com/shamilajeewantha/DiTra.git
cd DITRA
```

2. **Create and activate a conda environment (better with conda):**

```bash
conda create --name <my-env>
conda create <my-env>
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## 🚀 Running the Server

1. **Start the Python server:**

```bash
cd .\ui\
python .\server.py
```

2. **Serve with DIART (in separate terminal):**

```bash
diart.serve --host 0.0.0.0 --port 7007
```

This will start the DIART server and bind it to all interfaces on port `7007`.

Open the browser and go to url: http://localhost:8000

