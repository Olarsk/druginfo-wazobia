
# Drug Information Assistant 🚀

An AI-powered multilingual drug information assistant for healthcare professionals and the public.

## Features

- Retrieves verified drug data from BNF-84, EMDEX, and PTHB9 (see `data/`)
- Uses OpenAI ChatGPT to enhance and summarize responses
- Multilingual support: Yoruba, Igbo, Hausa (translation and TTS)
- Interactive **Streamlit app** for easy querying
- Text-to-speech (TTS) for local Nigerian languages

## 🛠 Installation
- Connect your vector data source in .env (Pinecone API)
```bash
git clone https://github.com/Olarsk/druginfo-wazobia.git
cd druginfo-wazobia
pip install -r requirements.txt
```
- update your .env as necessary
  
## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run app/main.py
```

## 📁 Project Structure

- `app/` - Streamlit UI, chat, audio, and translation modules
- `backend/` - Retrieval-augmented generation (RAG) and Pinecone vector DB client
- `models/` - GPT and translation models
- `utils/` - Config and helper utilities
- `data/` - Drug data sources (BNF, EMDEX, PTHB9)

## 📝 Example Query

> How many tablets of 500mg paracetamol should an adult take?

## 📦 Requirements

See `requirements.txt` for dependencies (OpenAI, Streamlit, Pinecone, Transformers, etc).

## 🤝 Contributing

Pull requests and suggestions are welcome!

## 📄 License

MIT License
