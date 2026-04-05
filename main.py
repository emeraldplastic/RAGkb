from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil

CHROMA_PATH  = "./chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
UPLOAD_DIR   = "./uploaded_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    source_filter: str = None


def get_db():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="documents"
    )


def build_chain(retriever):
    llm = Ollama(model="llama3.2")

    prompt_template = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make anything up.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PDFPlumberLoader(file_path)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="documents"
    )
    db.add_documents(chunks)

    return {
        "message": f"Successfully uploaded {file.filename}",
        "pages": len(pages),
        "chunks": len(chunks)
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    db = get_db()

    search_kwargs = {"k": 3}
    if request.source_filter:
        search_kwargs["filter"] = {"source": os.path.join(UPLOAD_DIR, request.source_filter)}

    retriever = db.as_retriever(search_kwargs=search_kwargs)
    chain     = build_chain(retriever)
    answer    = chain.invoke(request.question)

    source_docs = retriever.invoke(request.question)
    sources = list(set([
        f"{os.path.basename(doc.metadata.get('source', ''))} p.{doc.metadata.get('page', '')}"
        for doc in source_docs
    ]))

    return {
        "answer": answer,
        "sources": sources
    }


@app.get("/documents")
def list_documents():
    files = []
    if os.path.exists(UPLOAD_DIR):
        files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    return {"documents": files}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    os.remove(file_path)

    return {"message": f"Deleted {filename}"}