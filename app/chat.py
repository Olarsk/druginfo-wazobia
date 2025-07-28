from backend.rag import rag_pipeline
from models.translate import translate_text, LANGUAGE_CODES

def chat_with_bot(user_query, target_lang):
    response = rag_pipeline(user_query)
    lang_code = LANGUAGE_CODES.get(target_lang, "eng_Latn")
    return translate_text(response, target_lang_code=lang_code)