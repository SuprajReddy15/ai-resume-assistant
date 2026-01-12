import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

docs = []
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"docs/{file}")
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="llama3")

print("\nAI ready. Ask questions. Type 'exit' to quit.\n")

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break

    docs = retriever.invoke(q)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use ONLY the context below to answer.

Context:
{context}

Question:
{q}
"""

    answer = llm.invoke(prompt)
    print("\n", answer, "\n")
