import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import init_chat_model


# -------------------------------------------------------------------
# Embedding model setup
# -------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Chroma vector store (persistent DB for embeddings)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Local persistence
)


# -------------------------------------------------------------------
# Indexing: Load + chunk + embed documents into vector DB
# -------------------------------------------------------------------
def index_documents(file_path: str):
    """
    Indexing step:
    1. Load raw text documents from file.
    2. Split into smaller chunks (for efficient embedding & retrieval).
    3. Generate embeddings for each chunk and store in the vector DB.
    """
    loader = TextLoader(file_path)
    raw_text = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=40, separator="\n\n")
    documents = text_splitter.split_documents(raw_text)

    # Store document embeddings into the DB
    db = vector_store.from_documents(documents, embeddings)
    return db


# -------------------------------------------------------------------
# Retrieval: Search for relevant context
# -------------------------------------------------------------------
def retrieve_context(query: str, db, k: int = 3):
    """
    Retrieval step:
    - Perform similarity search against the vector DB.
    - Return top-k most relevant chunks as context for the LLM.
    """
    results = db.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in results)


# -------------------------------------------------------------------
# Generation: Build prompt and invoke LLM
# -------------------------------------------------------------------
def generate_answer(query: str, context: str, llm):
    """
    Generation step:
    - Combine user query with retrieved context.
    - Pass to LLM for a contextualized answer.
    """
    prompt_template = """
    You are a helpful and empathetic customer support specialist. 
    Use the provided context to answer user queries. 
    If the context does not contain enough information, say so honestly.

    Context:
    {context}

    Question:
    {question}

    When you don't the information available in context, say I don't know the answer
    """
    prompt = prompt_template.replace("{context}", context).replace("{question}", query)
    return llm.invoke(prompt).content


# -------------------------------------------------------------------
# Main driver
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure API key for Google Gemini
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = input("Enter API key for Google Gemini: ")

    # Initialize Google Gemini LLM
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    # Index documents from file
    db = index_documents("user-manual.txt")

    print("RAG Engine started ✅. Type your question or 'stop' to quit.\n")

    while True:
        user_query = input(">> ").strip()
        if user_query.lower() == "stop":
            print("RAG Engine stopped ❌.")
            break

        if not user_query:
            continue

        # Retrieve relevant context
        context = retrieve_context(user_query, db)

        # Generate LLM answer
        answer = generate_answer(user_query, context, llm)

        print("\n=== Answer ===")
        print(answer)
        print("\n-----------------\n")
