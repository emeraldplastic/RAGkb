from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "./chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def load_db():
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="documents"
    )
    return db

def build_chain(db, source_filter=None):
    llm = Ollama(model="llama3.2")

    search_kwargs = {"k": 3}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    retriever = db.as_retriever(search_kwargs=search_kwargs)

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

    return chain, retriever

def ask(chain, retriever, question):
    print(f"\nQuestion: {question}")
    print("Thinking...\n")

    answer = chain.invoke(question)
    print(f"Answer:\n{answer}")

    print("\nSources:")
    docs = retriever.invoke(question)
    for doc in docs:
        print(f"  - {doc.metadata.get('source')} p.{doc.metadata.get('page')}")
if __name__ == "__main__":
    db = load_db()

    print("Available filters: enter a filename to search only that file, or press Enter to search all.")
    source_filter = input("Filter by file (or Enter for all): ").strip()
    source_filter = source_filter if source_filter else None

    chain, retriever = build_chain(db, source_filter=source_filter)

    print("\nRAG chain ready. Type 'quit' to exit.")
    while True:
        question = input("\nYou: ")
        if question.lower() == "quit":
            break
        ask(chain, retriever, question)