import streamlit as st
from app.chat import chat_with_bot
from app.audio import spitch_tts
from models.translate import translate_text, LANGUAGE_CODES

# Set page config for wider layout
st.set_page_config(layout="wide")

# --- Centralized Header ---
st.markdown("""
    <h1 style='text-align: center;'>Welcome to DrugInfo WaZoBia</h1>
    <p style='text-align: center;'><em>Your multilingual drug information chatbot with multilingual support.</em></p>
    <hr>
""", unsafe_allow_html=True)

lang_mapping = {"Yoruba": "yor_Latn", "Igbo": "ibo_Latn", "Hausa": "hau_Latn"}
spitch_voices = {
    "yoruba": {"Sade (Yoruba)": ("yo", "sade"), "Funmi (Yoruba)": ("yo", "funmi")},
    "igbo": {"Obinna (Igbo)": ("ig", "obinna"), "Ngozi (Igbo)": ("ig", "ngozi")},
    "hausa": {"Hasan (Hausa)": ("ha", "hasan"), "Amina (Hausa)": ("ha", "amina")},
}


# --- Side-by-side Query and Response ---
col_query, col_sep, col_response = st.columns([5, 0.2, 5])
with col_query:
    query = st.text_area("Please enter your medical query:", "", height=120, key="query_input")
    send_query = st.button("Send Query", key="get_response_btn")
with col_sep:
    st.markdown("<div style='height: 180px; border-left: 3px solid #888; margin: 0 auto;'></div>", unsafe_allow_html=True)
with col_response:
    llm_response_placeholder = st.empty()
    regen_response = st.button("Regenerate Response", key="regen_response_btn")

if "llm_output" not in st.session_state:
    st.session_state["llm_output"] = ""
if "translated_output" not in st.session_state:
    st.session_state["translated_output"] = ""

if send_query and query:
    with st.spinner("Generating response..."):
        llm_output = chat_with_bot(query, "en")
        st.session_state["llm_output"] = llm_output
        st.session_state["translated_output"] = ""  # Reset translation if new query
llm_response_placeholder.text_area("Response", value=st.session_state["llm_output"], height=120)
if regen_response and query:
    with st.spinner("Regenerating response..."):
        llm_output = chat_with_bot(query, "en")
        st.session_state["llm_output"] = llm_output
        st.session_state["translated_output"] = ""  # Reset translation if new query

# --- Translation Section ---
st.markdown("Would you like to translate the output to a Nigerian language? Select language below:")
selected_translation_lang = st.selectbox("Translation Language", ["", *lang_mapping.keys()], key="translation_lang")

# Centralize Get translation button
get_translation_col = st.columns([3, 2, 3])
with get_translation_col[1]:
    get_translation = st.button("Get translation", key="get_translation_btn")

if get_translation and st.session_state["llm_output"] and selected_translation_lang:
    with st.spinner("Translating response..."):
        lang_code = lang_mapping[selected_translation_lang]
        translated = translate_text(st.session_state["llm_output"], target_lang_code=lang_code)
        st.session_state["translated_output"] = translated

st.markdown("Translated Drug info")
st.text_area("Translated Drug info", value=st.session_state["translated_output"], height=120, label_visibility="collapsed")

# Centralize Regenerate translation button
regen_translation_col = st.columns([3, 2, 3])
with regen_translation_col[1]:
    regen_translation = st.button("Regenerate translation", key="regen_translation_btn")

if regen_translation and st.session_state["llm_output"] and selected_translation_lang:
    with st.spinner("Regenerating translation..."):
        lang_code = lang_mapping[selected_translation_lang]
        translated = translate_text(st.session_state["llm_output"], target_lang_code=lang_code)
        st.session_state["translated_output"] = translated

# --- Speech Output Section ---
st.markdown("Interested in speech output?")
col1, col2 = st.columns(2)
with col1:
    selected_tts_lang = st.selectbox("Select language", ["", "yoruba", "igbo", "hausa"], key="tts_lang")
with col2:
    available_spitch_voices = spitch_voices.get(selected_tts_lang, {})
    selected_voice = st.selectbox("Select voice artist:", ["", *available_spitch_voices.keys()], key="tts_voice")

# Centralize Generate voice output button
tts_btn_col = st.columns([3, 2, 3])
with tts_btn_col[1]:
    tts_btn = st.button("Generate voice output", key="tts_btn")

if tts_btn and st.session_state["translated_output"] and selected_tts_lang and selected_voice:
    lang_code, voice_name = spitch_voices[selected_tts_lang][selected_voice]
    audio_bytes = spitch_tts(st.session_state["translated_output"], lang_code, voice_name)
    st.session_state["audio_bytes"] = audio_bytes

if "audio_bytes" in st.session_state:
    st.audio(st.session_state["audio_bytes"], format="audio/wav")
