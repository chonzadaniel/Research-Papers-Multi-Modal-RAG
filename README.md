# 🧠 Multimodal RAG App for Research Paper Exploration

This Streamlit-powered application enables **researchers, developers, and AI enthusiasts** to explore cutting-edge LLM and attention mechanism research papers using **Retrieval-Augmented Generation (RAG)** enhanced with **web search integration**. The system loads academic papers (PDFs), extracts and embeds their content using OpenAI models, and allows users to query them via a dynamic conversational interface.


# Features

- ✅ **Multimodal RAG**: Handles PDF documents with both text and tables
- ✅ **Integrated Web Search**: Extends RAG context with fresh knowledge from the internet
- ✅ **OpenAI Embeddings + LLM**: Supports OpenAI for both vector creation and generation
- ✅ **ChromaDB as Vector Store**: Fast and efficient local vector database
- ✅ **Streamlit UI**: Simple and interactive chat-like interface
- ✅ **PDF Research Sources**: GPT-4, Mistral 7B, Gemini, Attention Is All You Need, InstructGPT
- ✅ **Retrieval Reference**: Extend with document citations and source highlighting
- ✅ **Local Deployment Ready**


# File Structure

```bash
├── openai_chromadb_rag_app.py       # Main Streamlit App
├── requirements.txt                 # Python dependencies
├── /data                           # Folder containing uploaded PDF research papers
│   ├── attention_paper.pdf
│   ├── gemini_paper.pdf
│   ├── gpt4.pdf
│   ├── instructgpt.pdf
│   └── mistral_paper.pdf
├── /vector_store                   # Persisted ChromaDB index (after first run)
│   ├── chroma.sqlite
│   └── index/
```


# How It Works

1. **Data Ingestion**  
   PDFs are loaded, parsed (text and tables), and chunked for embedding.

2. **Embedding + Indexing**  
   Embeddings are generated using OpenAI models and stored in ChromaDB.

3. **Augmented Retrieval**  
   Chunks relevant to the user query are retrieved via vector search. Optionally, live web results are also fetched and fused into context.

4. **Response Generation**  
   Retrieved knowledge (PDF + web) is fed into OpenAI LLMs to generate grounded, accurate responses.

5. **Streamlit UI**  
   Offers a clean, scrollable chat interface for querying and exploring results.

# 📦 Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/multimodal-rag-app.git
cd multimodal-rag-app

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run openai_chromadb_rag_app.py
```

# 🧠 Example Queries You Can Ask

- “What architecture was proposed in the Mistral paper?”
- “Summarize the GPT-4 training methodology.”
- “Compare Gemini’s retrieval techniques with InstructGPT.”
- “What is the attention mechanism described in the 2017 Transformer paper?”

# Limitations
- Only PDF research papers are supported in the current version
- Web search is basic and may need rate-limiting or proxy setup
- OpenAI API is required (ensure your keys are loaded in `.env`)

# 🤝 Acknowledgments
- [OpenAI](https://openai.com/) – for embeddings & chat models
- [LangChain](https://www.langchain.com/) – used under the hood for RAG workflows
- [ChromaDB](https://www.trychroma.com/) – local vector store
- Research papers from [arXiv.org](https://arxiv.org/), [Google DeepMind](https://deepmind.google/) & [OpenAI](https://openai.com/research)

# Future Outlook
- Add support for images/diagrams (e.g., via BLIP or CLIP)
- Integrate with LangSmith or WandB for observability
- Enable user authentication (multi-user chat interface)
- Deploy on HuggingFace Spaces or Streamlit Cloud

# License

MIT License — see `LICENSE` file for details.
