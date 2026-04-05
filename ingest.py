from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

CHROMA_PATH = "./chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} doesn't exist.")
        return []
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"Loaded {len(pages)} pages from {pdf_path}")
    return pages


def split_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def store_in_chromadb(chunks, doc_id):
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="documents"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    print(f"Database saved to: {CHROMA_PATH}")
    return db


def test_search(query, n_results=3):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="documents"
    )
    results = db.similarity_search_with_score(query, k=n_results)
    print(f"\nTop {n_results} results for: '{query}'\n")
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] Score: {score:.3f}")
        print(f"    Source: {doc.metadata.get('source')} p.{doc.metadata.get('page')}")
        print(f"    Text: {doc.page_content[:200]}...")
        print()


if __name__ == "__main__":
    PDF_PATH = "synopsis1.pdf"   # change this to your PDF filename

    pages  = load_pdf(PDF_PATH)
    chunks = split_into_chunks(pages)
    store_in_chromadb(chunks, doc_id="doc_001")
    test_search("What is the main topic of this document?")