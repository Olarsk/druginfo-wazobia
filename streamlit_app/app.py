import streamlit as st
from models.gpt_model import chatgpt_generate

st.title("💊 Drug Information Assistant (Multilingual)")
st.write("Ask any drug-related question!")

# Language selection for final output
lang_options = {
    "English": "en",
    "Hausa": "ha",
    "Yoruba": "yo",
    "Igbo": "ig",
    "French": "fr"
}
selected_lang = st.selectbox("Choose your final response language:", list(lang_options.keys()))

query = st.text_input("Enter your query in English:")
if query:
    with st.spinner("Fetching information..."):
        response = chatgpt_generate(query, final_lang=lang_options[selected_lang])
    st.success("✅ Response Generated!")
    st.write(response)
