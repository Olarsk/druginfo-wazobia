import os
import openai
from data.chromadb_store import retrieve_drug_info
from utils.config import load_env

# Load environment variables
api_key = load_env("OPENAI_API_KEY")
client = openai.Client(api_key=api_key)

def chatgpt_generate(prompt, model="gpt-3.5-turbo"):
    """Generates a response from OpenAI's GPT model, including similarity search results."""

    # Retrieve relevant drug info from ChromaDB
    similarity_result = retrieve_drug_info(prompt)

    # Construct the system prompt dynamically with retrieved info
    system_prompt = f"""
    You are a highly intelligent summarization expert with deep expertise in pharmaceuticals and clinical drug information.
    You have been provided with multiple pieces of data from reputable sources regarding the following inquiry:

    Retrieved Information:
    {similarity_result}

    Your task is to carefully analyze all the provided information and synthesize a clear, concise, and highly accurate response
    that captures all the essential details, including any specific dosage recommendations or instructions if mentioned.
    Do not include any extraneous details—only provide the necessary pharmaceutical information in one coherent paragraph.

    At the end of your response, explicitly mention the sources used in this format:
    
    "Sources Used: BNF-84, EMDEX (or other relevant sources from the retrieved information)"
    """

    # Call OpenAI API with the formatted prompt
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )


    formatted_response = response.choices[0].message.content.strip()
    return formatted_response
