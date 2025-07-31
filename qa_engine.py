import fitz  # PyMuPDF
import os
import re

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load and clean a PDF (smaller memory footprint)
def load_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Clean up
    full_text = re.sub(r"http\S+", "", full_text)
    full_text = re.sub(r"Page \d+", "", full_text)
    full_text = re.sub(r"\s{2,}", " ", full_text)

    # Break into smaller chunks (≤ 300 words)
    sentences = re.split(r'(?<=[.!?]) +', full_text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) < 300:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())

    return [c for c in chunks if len(c) > 100]

# Setup vector store
def setup_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, embeddings)

# Load small local model

def load_local_llm():
    model_name = "google/flan-t5-small"  # uses less memory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

# Initialization
print("🔄 Loading PDFs and preparing vector store...")
pdf_folder = "docs"
all_chunks = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_folder, filename)
        print(f"📄 Reading: {filename}")
        all_chunks.extend(load_pdf(path))

print(f"Total Chunks: {len(all_chunks)}")

vector_store = setup_vector_store(all_chunks)
llm = load_local_llm()
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Exposed to app.py
def ask_robotics_question(question):
    result = qa_chain(question)

    print("🔍 Retrieved Chunks:")
    for doc in result['source_documents']:
        print(doc.page_content[:500])

    sources = "\n\n".join([doc.page_content for doc in result['source_documents']])
    return f"\U0001F4C4 **Context:**\n\n{sources}\n\n\U0001F916 **Answer:**\n\n{result['result']}"
