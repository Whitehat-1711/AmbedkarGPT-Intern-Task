# AmbedkarGPT-Intern-Task

A RAG (Retrieval-Augmented Generation) based Q&A system that answers questions about Dr. B.R. Ambedkar's speech excerpt from "Annihilation of Caste" using LangChain, ChromaDB, and Ollama.

## 📋 Overview

This system implements a complete RAG pipeline that:
- Loads text from a speech by Dr. B.R. Ambedkar
- Splits the text into manageable chunks
- Creates embeddings using HuggingFace models
- Stores embeddings in ChromaDB vector database
- Retrieves relevant context based on user questions
- Generates answers using Ollama's Mistral 7B model

## 🛠️ Technical Stack

- **Framework**: LangChain
- **Vector Database**: ChromaDB (local, persistent)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Ollama with Mistral 7B
- **Language**: Python 3.8+

## 📦 Prerequisites

### 1. Install Python 3.8+
Ensure you have Python 3.8 or higher installed on your system.

### 2. Install Ollama
Download and install Ollama from [https://ollama.ai](https://ollama.ai)

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from the Ollama website.

### 3. Pull Mistral Model
After installing Ollama, pull the Mistral 7B model:
```bash
ollama pull mistral
```

Verify the installation:
```bash
ollama list
```

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── main.py              # Command-line RAG system
├── app.py               # Streamlit web interface (NEW!)
├── speech.txt           # Dr. Ambedkar's speech excerpt
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── chroma_db/          # Vector database (created on first run)
```

## ▶️ Usage

### Option 1: Run with Streamlit (Recommended - Interactive UI)
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: Run Command-Line Version
```bash
python main.py
```

### Example Interaction (Streamlit)
```
================================================================================
AmbedkarGPT - Q&A System
================================================================================
Loading document...
Splitting text into chunks...
Created 3 text chunks
Loading embedding model...
Creating vector store...
Initializing LLM...
Setting up QA chain...

================================================================================
System ready! You can now ask questions.
================================================================================

Enter your question (or 'quit' to exit): What is the real remedy according to the text?

Question: What is the real remedy according to the text?
--------------------------------------------------------------------------------
Answer: The real remedy is to destroy the belief in the sanctity of the shastras...
--------------------------------------------------------------------------------
```

### Sample Questions to Try
- What is the real remedy according to the text?
- What is the relationship between caste and shastras?
- What metaphor is used to describe social reform?
- What must people stop believing in?

### Exit the Program
Type `quit`, `exit`, or `q` to exit the program.

## 🔧 How It Works

### 1. Document Loading
The system loads `speech.txt` using LangChain's `TextLoader`.

### 2. Text Chunking
Text is split into chunks of 200 characters with 50 character overlap using `CharacterTextSplitter`.

### 3. Embedding Generation
Each chunk is converted to a vector embedding using the `sentence-transformers/all-MiniLM-L6-v2` model.

### 4. Vector Storage
Embeddings are stored in ChromaDB for efficient similarity search.

### 5. Question Processing
When a question is asked:
- The question is embedded using the same model
- Top 3 most similar chunks are retrieved
- Retrieved context + question are sent to Mistral 7B
- LLM generates a contextual answer

### 6. Answer Generation
Ollama's Mistral 7B model generates an answer based on the retrieved context.

## 🐛 Troubleshooting

### Issue: "Ollama not found"
**Solution**: Ensure Ollama is installed and the `ollama` command works in terminal.

### Issue: "Mistral model not found"
**Solution**: Run `ollama pull mistral` to download the model.

### Issue: Import errors
**Solution**: Ensure you're in the virtual environment and all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "No such file: speech.txt"
**Solution**: Ensure `speech.txt` is in the same directory as `main.py`.

### Issue: ChromaDB errors
**Solution**: Delete the `chroma_db` folder and run the program again to recreate it.

## 📝 Notes

- First run will be slower as it downloads the embedding model (~90MB)
- The `chroma_db` folder persists between runs for faster startup
- All components run locally with no API keys or accounts required
- The system is designed for demonstration and learning purposes

## 🤝 Contributing

This is an assignment project. For any questions, contact the hiring manager at kalpiksingh2005@gmail.com

## 📄 License

This project is created as part of the Kalpit Pvt Ltd AI Intern hiring assignment.

---

**Created by**: Nishant Gosavi  
**Assignment**: AI Intern Hiring - Phase 1  
**Company**: Kalpit Pvt Ltd, UK
