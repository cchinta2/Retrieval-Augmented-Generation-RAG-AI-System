# Retrieval-Augmented Generation (RAG) Engine  

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** using LangChain, Hugging Face embeddings, Chroma vector store, and Google Gemini (via `langchain.chat_models`).  

RAG combines two steps:  
1. **Retrieval** – Fetching relevant information from an external knowledge base (documents stored as embeddings in Chroma).  
2. **Generation** – Using an LLM to generate answers grounded in retrieved context.  

By combining retrieval + generation, this engine reduces hallucinations and ensures more **accurate, context-aware answers**.  

---

## 🚀 Features  
- Document indexing (load → split → embed → persist in Chroma DB).  
- Similarity-based retrieval for queries.  
- Answer generation using Google Gemini.  
- Interactive **command-line interface** for asking questions.  
- Graceful stop command (`stop`).  

---

## ⚙️ Setup & Run  

### 1. Clone the repository  
```bash
git clone https://github.com/cchinta2/Retrieval-Augmented-Generation-RAG-AI-System.git
```

### 2. Create a virutal environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies  

See `requirements.txt`. Install with:  
```bash
pip install -r requirements.txt
```

### 4. Run the program
```bash
python rag_engine.py
```