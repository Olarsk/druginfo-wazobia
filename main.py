import streamlit as st
from app.chat import chat_with_bot
from app.audio import spitch_tts
from models.translate import LANGUAGE_CODES

st.title("Welcome to DrugInfo WaZoBia")
st.markdown("""
<div style='text-align: left;'>
    <em>Your multilingual drug information chatbot with multilingual support.</em>
</div>
<br>
""", unsafe_allow_html=True)
st.markdown("---")
tts_disabled = "translated_text" not in st.session_state

lang_mapping = {"Yoruba": "yor_Latn", "Igbo": "ibo_Latn", "Hausa": "hau_Latn"}
spitch_voices = {
    "yoruba": {"Sade (Yoruba)": ("yo", "sade"), "Funmi (Yoruba)": ("yo", "funmi")},
    "igbo": {"Obinna (Igbo)": ("ig", "obinna"), "Ngozi (Igbo)": ("ig", "ngozi")},
    "hausa": {"Hasan (Hausa)": ("ha", "hasan"), "Amina (Hausa)": ("ha", "amina")},
}

# --- Query Section ---
query = st.text_area("Please enter your medical query:", "", height=120, key="query_input")

# --- Response Section ---
st.markdown("<b>Response:</b>", unsafe_allow_html=True)
response_placeholder = st.empty()

# --- Translation Section ---
st.markdown("Would you like to translate the output to a Nigerian language? Select language below:")
selected_translation_lang = st.selectbox("Translation Language", ["", *lang_mapping.keys()], key="translation_lang")

# --- Speech Output Section ---
st.markdown("Interested in speech output?")
col1, col2 = st.columns(2)
with col1:
    selected_tts_lang = st.selectbox("Select language", ["", "yoruba", "igbo", "hausa"], key="tts_lang")
with col2:
    available_spitch_voices = spitch_voices.get(selected_tts_lang, {})
    selected_voice = st.selectbox("Select voice artist:", ["", *available_spitch_voices.keys()], key="tts_voice")

# --- Main Logic ---
if query:
    if st.button("Get Response", key="get_response_btn"):
        with st.spinner("Generating response..."):
            response = chat_with_bot(query, lang_mapping.get(selected_translation_lang, ""))
            st.session_state["translated_text"] = response
        response_placeholder.text_area("Response", value=st.session_state["translated_text"], height=120)
    elif "translated_text" in st.session_state:
        response_placeholder.text_area("Response", value=st.session_state["translated_text"], height=120)

# Optionally, add TTS button if all fields are selected
if (
    "translated_text" in st.session_state and
    selected_tts_lang and
    selected_voice and
    selected_voice in spitch_voices.get(selected_tts_lang, {})
):
    if st.button("🔊 Play Speech Output", key="play_tts_btn"):
        lang_code, voice_name = spitch_voices[selected_tts_lang][selected_voice]
        audio_bytes = spitch_tts(st.session_state["translated_text"], lang_code, voice_name)
        st.audio(audio_bytes, format="audio/wav")
