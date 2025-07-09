import requests
import os

SPITCH_API_URL = os.getenv("SPITCH_API_URL")
SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
SPITCH_TTS_URL = os.getenv("SPITCH_TTS_URL")  # Add this to your .env if different from translation

def spitch_translate(text, target_lang):
    response = requests.post(
        SPITCH_API_URL,
        headers={"Authorization": f"Bearer {SPITCH_API_KEY}"},
        json={"text": text, "target_lang": target_lang}
    )
    response.raise_for_status()
    return response.json()["translation"]

def spitch_tts(text, lang_code):
    """Generate audio from text using Spitch TTS API."""
    response = requests.post(
        SPITCH_TTS_URL or SPITCH_API_URL,  # fallback to translation URL if TTS URL not set
        headers={"Authorization": f"Bearer {SPITCH_API_KEY}"},
        json={"text": text, "lang": lang_code}
    )
    response.raise_for_status()
    # Assume API returns a URL or base64-encoded audio
    return response.json().get("audio_url") or response.json().get("audio_base64") 