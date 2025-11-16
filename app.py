import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Disable ChromaDB telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Page configuration
st.set_page_config(
    page_title="AmbedkarGPT",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def setup_rag_system():
    """Initialize and setup the complete RAG pipeline - cached for performance"""
    
    try:
        # Step 1: Load the speech text file
        loader = TextLoader("speech.txt", encoding="utf-8")
        documents = loader.load()
        
        # Step 2: Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separator="\n"
        )
        chunks = text_splitter.split_documents(documents)
        
        # Step 3: Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Step 4: Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Step 5: Initialize Ollama LLM
        llm = Ollama(model="mistral")
        
        # Step 6: Create custom prompt for relevance checking
        prompt_template = """Use the following pieces of context from Dr. B.R. Ambedkar's speech to answer the question. 

IMPORTANT: Only answer questions that can be answered using the provided context. If the question is not related to the context or cannot be answered using the information provided, respond EXACTLY with: "I can only answer questions based on the provided speech by Dr. B.R. Ambedkar about caste, shastras, and social reform."

Context: {context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain, None
    
    except FileNotFoundError:
        return None, "❌ Error: speech.txt file not found. Please ensure it's in the same directory."
    except Exception as e:
        return None, f"❌ Error during setup: {str(e)}"

def main():
    # Header
    st.title("📚 AmbedkarGPT - Q&A System")
    st.markdown("*Ask questions about Dr. B.R. Ambedkar's speech on caste and social reform*")
    st.divider()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Setup RAG system
    with st.spinner("🔧 Initializing system... (This may take a moment on first run)"):
        qa_chain, error = setup_rag_system()
    
    if error:
        st.error(error)
        st.info("**Setup Instructions:**\n1. Ensure speech.txt is in the same directory\n2. Install Ollama and pull Mistral model\n3. Install all dependencies from requirements.txt")
        st.stop()
    
    st.success("✅ System ready!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("💭 Sample Questions")
        st.markdown("""
        - What is the real remedy according to Ambedkar?
        - What is the relationship between caste and shastras?
        - How does Ambedkar describe the work of social reform?
        - What must people stop believing in to get rid of caste?
        - Who is the real enemy mentioned in the speech?
        """)
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        This system uses:
        - **LangChain** for RAG pipeline
        - **ChromaDB** for vector storage
        - **HuggingFace** embeddings
        - **Ollama Mistral 7B** for generation
        """)
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "chunks" in message:
                with st.expander("📄 View Retrieved Chunks"):
                    for i, chunk in enumerate(message["chunks"], 1):
                        st.text_area(
                            f"Chunk {i}",
                            chunk,
                            height=100,
                            disabled=True,
                            key=f"chunk_{len(st.session_state.messages)}_{i}"
                        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the speech..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    result = qa_chain.invoke({"query": prompt})
                    
                    answer = result['result'].strip()
                    source_docs = result.get('source_documents', [])
                    
                    # Check for irrelevant questions
                    if "I can only answer questions based on the provided speech" in answer:
                        response = "⚠️ **Question Not Relevant**\n\nThis question cannot be answered using the provided speech content.\n\nPlease ask questions related to:\n- Caste system and shastras\n- Social reform\n- Dr. Ambedkar's views on these topics"
                        st.warning(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                    else:
                        # Display answer
                        st.markdown(answer)
                        
                        # Display retrieved chunks in expander
                        if source_docs:
                            with st.expander("📄 View Retrieved Chunks"):
                                for i, doc in enumerate(source_docs, 1):
                                    st.text_area(
                                        f"Chunk {i}",
                                        doc.page_content,
                                        height=100,
                                        disabled=True,
                                        key=f"chunk_current_{i}"
                                    )
                        
                        # Save to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "chunks": [doc.page_content for doc in source_docs]
                        })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()