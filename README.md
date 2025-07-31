# Thesis Q&A Chatbot 🤖📘

A lightweight Streamlit app that lets users ask questions about Darshan's master's thesis using a free local LLM.

## Features
- Ask about abstract, conclusion, methodology, etc.
- Uses Mistral-7B or any HuggingFace LLM (no API key needed)
- Uses FAISS vector search for fast retrieval
- Clean UI with Streamlit

## Setup Instructions

1. Clone the project or download files
2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the app:
```
streamlit run app.py
```

4. Ask your questions!

## Notes
- Ensure your system can run HuggingFace models (better with GPU)
- Place your thesis file in the same folder named exactly as: `Thesis_finalreport_DarshanBR.pdf`
