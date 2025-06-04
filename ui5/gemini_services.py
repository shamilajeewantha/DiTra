from google import genai
from google.genai import types
import logging

# Set up the second logger for a second log
logger3 = logging.getLogger("logger3")
logger3.setLevel(logging.INFO)
file_handler3 = logging.FileHandler("gemini_log.txt")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler3.setFormatter(formatter)
logger3.addHandler(file_handler3)


def load_api_key(filepath="api_key.txt"):
    with open(filepath, "r") as f:
        return f.read().strip()

def load_transcript(filepath="received_audio.srt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate(mode="offline", full_transcript=""):
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    # Read the original transcript
    if mode=="offline":
        transcript_text = load_transcript()
            # Your prompt combined with transcript text
        prompt = (
            "In the speaker diarization transcript above, some words are potentially assigned to the wrong speaker. "
            "Please correct those words and move them to the right speaker. You may need to re-label the speaker id of entire segments. "
            "Merge adjacent segments if they belong to the same speaker such that adjacent segments always belong to different speakers. "
            "The order of the text is correct. Directly show the corrected transcript without explaining what changes were made or why you made those changes.\n\n"
            + transcript_text
        )

    else:
        transcript_text = full_transcript
            # Your prompt combined with transcript text
        prompt = (
            "In the speaker diarization transcript above, some words are potentially assigned to the wrong speaker. "
            "Please correct those words and move them to the right speaker. Assign all unknown words to an appropriate speaker."
            "The order of the text is correct. Directly show the corrected lines in the exact same format without explaining what changes were made or why you made those changes. Make sure the result is not a code block. \n\n"
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
    if mode == "offline":
        with open("diarization_result.txt", "w", encoding="utf-8") as f:
            f.write(output_text)
        print("Corrected transcript saved to diarization_result.txt.")

    else:
        logger3.info(f"Gemini output:\n{output_text}\n\n\n")

    return output_text
