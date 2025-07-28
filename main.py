
import streamlit as st
from app.chat import chat_with_bot
from app.audio import spitch_tts
from models.translate import translate_text, LANGUAGE_CODES

st.title("Welcome to DrugInfo WaZoBia")
st.markdown("""
<div style='text-align: left;'>
    <em>Your multilingual drug information chatbot with multilingual support.</em>
</div>
<br>
""", unsafe_allow_html=True)
st.markdown("---")

lang_mapping = {"Yoruba": "yor_Latn", "Igbo": "ibo_Latn", "Hausa": "hau_Latn"}
spitch_voices = {
    "yoruba": {"Sade (Yoruba)": ("yo", "sade"), "Funmi (Yoruba)": ("yo", "funmi")},
    "igbo": {"Obinna (Igbo)": ("ig", "obinna"), "Ngozi (Igbo)": ("ig", "ngozi")},
    "hausa": {"Hasan (Hausa)": ("ha", "hasan"), "Amina (Hausa)": ("ha", "amina")},
}


# --- Query Section ---
query = st.text_area("Please enter your medical query:", "", height=80, key="query_input")
send_query = st.button("Send Query", key="get_response_btn")

# --- LLM Response Section ---
st.markdown("<b>Response:</b>", unsafe_allow_html=True)
llm_response_placeholder = st.empty()
if "llm_output" not in st.session_state:
    st.session_state["llm_output"] = ""
if "translated_output" not in st.session_state:
    st.session_state["translated_output"] = ""

if send_query and query:
    with st.spinner("Generating response..."):
        llm_output = chat_with_bot(query, "en")
        st.session_state["llm_output"] = llm_output
        st.session_state["translated_output"] = ""  # Reset translation if new query
        
llm_response_placeholder.text_area("Response", value=st.session_state["llm_output"], height=200)
regen_response = st.button("Regenerate Response", key="regen_response_btn")
if regen_response and query:
    with st.spinner("Regenerating response..."):
        llm_output = chat_with_bot(query, "en")
        st.session_state["llm_output"] = llm_output
        st.session_state["translated_output"] = ""  # Reset translation if new query

# --- Translation Section ---
st.markdown("Would you like to translate the output to a Nigerian language? Select language below:")
selected_translation_lang = st.selectbox("Translation Language", ["", *lang_mapping.keys()], key="translation_lang")

if st.button("Regenerate Response", key="translate_btn") and st.session_state["llm_output"] and selected_translation_lang:
    with st.spinner("Translating response..."):
        lang_code = lang_mapping[selected_translation_lang]
        translated = translate_text(st.session_state["llm_output"], target_lang_code=lang_code)
        st.session_state["translated_output"] = translated

st.markdown("Translated Drug info")
st.text_area("", value=st.session_state["translated_output"], height=120)

# --- Speech Output Section ---
st.markdown("Interested in speech output?")
col1, col2 = st.columns(2)
with col1:
    selected_tts_lang = st.selectbox("Select language", ["", "yoruba", "igbo", "hausa"], key="tts_lang")
with col2:
    available_spitch_voices = spitch_voices.get(selected_tts_lang, {})
    selected_voice = st.selectbox("Select voice artist:", ["", *available_spitch_voices.keys()], key="tts_voice")

if st.button("Generate voice output", key="tts_btn") and st.session_state["translated_output"] and selected_tts_lang and selected_voice:
    lang_code, voice_name = spitch_voices[selected_tts_lang][selected_voice]
    audio_bytes = spitch_tts(st.session_state["translated_output"], lang_code, voice_name)
    st.session_state["audio_bytes"] = audio_bytes

if "audio_bytes" in st.session_state:
    st.audio(st.session_state["audio_bytes"], format="audio/wav")
