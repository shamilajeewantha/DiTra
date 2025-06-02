import os
from google import genai
from google.genai import types

def load_api_key(filepath="api_key.txt"):
    with open(filepath, "r") as f:
        return f.read().strip()

def load_transcript(filepath="received_audio.srt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate():
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    # Read the original transcript
    transcript_text = load_transcript()

    # Your prompt combined with transcript text
    prompt = (
        "In the speaker diarization transcript above, some words are potentially assigned to the wrong speaker. "
        "Please correct those words and move them to the right speaker. You may need to re-label the speaker id of entire segments. "
        "Merge adjacent segments if they belong to the same speaker such that adjacent segments always belong to different speakers. "
        "The order of the text is correct. Directly show the corrected transcript without explaining what changes were made or why you made those changes.\n\n"
        + transcript_text
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")
    model = "gemini-2.5-flash-preview-05-20"

    # Collect output
    output_text = ""

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        output_text += chunk.text

    # Save to file
    with open("diarization_result.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    print("Corrected transcript saved to diarization_result.txt.")

if __name__ == "__main__":
    generate()
