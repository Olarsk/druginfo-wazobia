import streamlit as st
from src.models.gpt_model import chatgpt_generate

st.title("💊 Drug Information Assistant")
st.write("Ask any drug-related question, and I'll retrieve data from verified sources!")

query = st.text_input("Enter your query:")
if query:
    with st.spinner("Fetching information..."):
        response = chatgpt_generate(query)
    st.success("✅ Response Generated!")
    st.write(response)
