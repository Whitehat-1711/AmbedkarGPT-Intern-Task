from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

def setup_rag_system():
    """Initialize and setup the complete RAG pipeline"""
    
    # Step 1: Load the speech text file
    print("Loading document...")
    try:
        loader = TextLoader("speech.txt", encoding="utf-8")
        documents = loader.load()
    except FileNotFoundError:
        print("❌ Error: speech.txt file not found in the current directory.")
        print("Please ensure speech.txt is in the same folder as main.py")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        exit(1)
    
    # Step 2: Split text into chunks for better retrieval
    print("Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    
    # Step 3: Create embeddings using HuggingFace model
    print("Loading embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        print("This may be the first run. The model will be downloaded (~90MB).")
        exit(1)
    
    # Step 4: Create and populate Chroma vector store
    print("Creating vector store...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        exit(1)
    
    # Step 5: Initialize Ollama LLM with Mistral
    print("Initializing LLM...")
    try:
        llm = Ollama(model="mistral")
    except Exception as e:
        print(f"❌ Error initializing Ollama: {e}")
        print("Please ensure Ollama is installed and Mistral model is downloaded:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull Mistral: ollama pull mistral")
        exit(1)
    
    # Step 6: Create RetrievalQA chain with custom prompt
    print("Setting up QA chain...")
    
    # Custom prompt to enforce relevance checking
    prompt_template = """Use the following pieces of context from Dr. B.R. Ambedkar's speech to answer the question. 

IMPORTANT: Only answer questions that can be answered using the provided context. If the question is not related to the context or cannot be answered using the information provided, respond EXACTLY with: "I can only answer questions based on the provided speech by Dr. B.R. Ambedkar about caste, shastras, and social reform."

Context: {context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def ask_question(qa_chain, question):
    """Query the RAG system with a question"""
    print(f"\nQuestion: {question}")
    print("=" * 80)
    
    try:
        result = qa_chain.invoke({"query": question})
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        return None
    
    # Check if any relevant chunks were retrieved
    if not result.get('source_documents'):
        print("⚠️ No relevant information found in the document.")
        print("Try rephrasing your question or ask about topics mentioned in the speech.")
        return None
    
    # Check if the answer indicates irrelevance
    answer = result['result'].strip()
    if "I can only answer questions based on the provided speech" in answer:
        print("\n⚠️ Question Not Relevant")
        print("-" * 80)
        print("This question cannot be answered using the provided speech content.")
        print("Please ask questions related to:")
        print("  • Caste system and shastras")
        print("  • Social reform")
        print("  • Dr. Ambedkar's views on these topics")
        print("=" * 80)
        return None
    
    # Display retrieved chunks
    print("\n📄 Retrieved Chunks:")
    print("-" * 80)
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\nChunk {i}:")
        print(f"{doc.page_content}")
        print("-" * 40)
    
    # Display the generated answer
    print("\n💡 Answer:")
    print("-" * 80)
    print(f"{answer}")
    print("=" * 80)
    return result

def main():
    """Main execution function"""
    print("=" * 80)
    print("AmbedkarGPT - Q&A System")
    print("=" * 80)
    
    # Setup the RAG system
    try:
        qa_chain = setup_rag_system()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user.")
        exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        exit(1)
    
    print("\n" + "=" * 80)
    print("System ready! You can now ask questions.")
    print("=" * 80)
    
    # Display sample questions
    print("\n💭 Sample questions you can ask:")
    print("  • What is the real remedy according to Ambedkar?")
    print("  • What is the relationship between caste and shastras?")
    print("  • How does Ambedkar describe the work of social reform?")
    print("  • What must people stop believing in to get rid of caste?")
    print("  • Who is the real enemy mentioned in the speech?")
    print("\nType 'quit' to exit.\n")
    
    # Interactive Q&A loop
    while True:
        try:
            user_question = input("Enter your question: ").strip()
            
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\n✅ Thank you for using AmbedkarGPT!")
                break
            
            if not user_question:
                print("⚠️ Please enter a valid question.")
                continue
            
            ask_question(qa_chain, user_question)
            
        except KeyboardInterrupt:
            print("\n\n✅ Thank you for using AmbedkarGPT!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()