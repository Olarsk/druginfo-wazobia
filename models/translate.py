"""
NLLB-200 Translation Module

Provides translation using the facebook/nllb-200-distilled-600M model.
Supports Hausa, Yoruba, Igbo, French, and English by default.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Load tokenizer & model (cached on first use)
tokenizer_nllb = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME).to(DEVICE)

# Supported language codes in NLLB-200
LANGUAGE_CODES = {
    "en": "eng_Latn",   # English
    "ha": "hau_Latn",   # Hausa
    "yo": "yor_Latn",   # Yoruba
    "ig": "ibo_Latn",   # Igbo
    "fr": "fra_Latn",   # French
    # Add more as needed
}

def get_lang_code(lang: str) -> str:
    """Get NLLB language code from short language key (e.g., 'yo')."""
    if lang not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language: {lang}")
    return LANGUAGE_CODES[lang]

def translate_text(text, target_lang_code="eng_Latn"):
    """
    Translate text to the target language using NLLB-200.
    Args:
        text (str): The text to translate.
        target_lang_code (str): NLLB language code (e.g., 'yor_Latn').
    Returns:
        str: Translated text.
    """
    if target_lang_code == "eng_Latn":  # No translation needed for English
        return text

    inputs = tokenizer_nllb(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    forced_bos_token_id = tokenizer_nllb.convert_tokens_to_ids(target_lang_code)

    translated_tokens = model_nllb.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=500,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True,
        num_beams=5
    )

    return tokenizer_nllb.decode(translated_tokens[0], skip_special_tokens=True)
