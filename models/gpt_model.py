import os
import openai
from models.translate import translate_text, LANGUAGE_CODES
from utils.config import load_env

api_key = load_env("OPENAI_API_KEY")
client = openai.Client(api_key=api_key)

def chatgpt_generate(prompt, context=None, model="gpt-4o", final_lang="en"):
    system_prompt = f"""
    You are a highly intelligent summarization expert with deep expertise in pharmaceuticals and clinical drug information.
    You have been provided with multiple pieces of data from reputable sources regarding the following inquiry:

    Retrieved Information:
    {context}

    Your task is to carefully analyze all the provided information and synthesize a clear, concise, and highly accurate response
    that captures all the essential details, including any specific dosage recommendations or instructions if mentioned.
    Do not include any extraneous details—only provide the necessary pharmaceutical information in one coherent paragraph.

    At the end of your response, explicitly mention the sources used in this format:
    "Sources Used: BNF-84, EMDEX (or other relevant sources from the retrieved information)"
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    gpt_response = response.choices[0].message.content.strip()
    final_lang_code = LANGUAGE_CODES.get(final_lang, "eng_Latn")
    translated_response = translate_text(gpt_response, target_lang_code=final_lang_code)
    return translated_response
