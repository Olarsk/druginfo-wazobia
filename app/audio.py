from spitch import Spitch
import os

def spitch_tts(text, language, voice, output_path="spitch_output.mp3"):
    """
    Generate audio from text using the Spitch Python client.
    Returns: path to the generated MP3 file.
    """
    spitch_api_key = os.getenv("SPITCH_API_KEY")
    client = Spitch(api_key=spitch_api_key) if spitch_api_key else Spitch()
    with open(output_path, "wb") as f:
        response = client.speech.generate(
            text=text,
            language=language,
            voice=voice
        )
        f.write(response.read())
    return output_path 