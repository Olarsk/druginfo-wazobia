import streamlit as st
from app.chat import chat_with_bot
from app.audio import spitch_tts
from models.translate import LANGUAGE_CODES

st.title("💊 Multilingual Drug Info Chatbot")

lang_mapping = {"Yoruba": "yor_Latn", "Igbo": "ibo_Latn", "Hausa": "hau_Latn"}
spitch_voices = {
    "yoruba": {"Sade (Yoruba)": ("yo", "sade"), "Funmi (Yoruba)": ("yo", "funmi")},
    "igbo": {"Obinna (Igbo)": ("ig", "obinna"), "Ngozi (Igbo)": ("ig", "ngozi")},
    "hausa": {"Hasan (Hausa)": ("ha", "hasan"), "Amina (Hausa)": ("ha", "amina")},
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Enter Your Medical Query")
    query = st.text_area("", "How many tablets of 500mg paracetamol should an adult take?")
    st.markdown("### 🌍 Select Language for Translation")
    selected_translation_lang = st.selectbox("", list(lang_mapping.keys()))

with col2:
    st.markdown("### 🗣️ Choose TTS Language & Voice")
    selected_tts_lang = st.selectbox("", ["yoruba", "igbo", "hausa"])
    available_spitch_voices = spitch_voices[selected_tts_lang]
    selected_voice = st.selectbox("", list(available_spitch_voices.keys()))

st.markdown("---")

if st.button("🚀 Generate Summary & Translate"):
    with st.spinner("🔄 Generating response..."):
        response = chat_with_bot(query, lang_mapping[selected_translation_lang])
        st.session_state["translated_text"] = response
    st.success("✅ Translation Complete!")
    st.markdown(f"### 📖 Translated Text ({selected_translation_lang})")
    st.info(st.session_state["translated_text"])

tts_disabled = "translated_text" not in st.session_state

if st.button("🎙️ Convert to Speech", disabled=tts_disabled):
    if tts_disabled:
        st.error("⚠️ Please generate a translation first!")
    else:
        with st.spinner("🔄 Generating Speech with Spitch..."):
            language_code, voice_id = available_spitch_voices[selected_voice]
            audio_path = spitch_tts(
                st.session_state["translated_text"],
                language=language_code,
                voice=voice_id
            )
            st.audio(audio_path, format="audio/mp3")
        st.success("✅ Speech Generated & Played!") 