import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main > div {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reasoning-box {
        background-color: #DFDFDFFF;
        border-left: 4px solid #6c757d;
        color: #111111FF;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border-radius: 0.5rem;
    }
    .answer-box {
        background-color: #DFDFDFFF;
        color: #111111FF;
        border-left: 4px solid #0969da;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    ### How to use
    1. Upload a PDF document
    2. Ask questions about the document
    
    ### Features
    - PDF document analysis
    - Question answering
    """)

# Main content
st.title("📚 Document Q&A Assistant")
st.markdown("### Upload your document and ask questions")

# Groq API Key (replace with your actual key)
GROQ_API_KEY = "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"

def setup_qa_chain(pdf_content):
    # Save temporary file
    temp_file_path = "temp_document.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(pdf_content)
    
    # Load and process document
    loader = PDFPlumberLoader(temp_file_path)
    docs = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    documents = text_splitter.split_documents(docs)
    
    # Create vector store with HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        model_name="deepseek-r1-distill-llama-70b"
    )
    
    # Set up QA chain
    prompt = """
    Based on the provided context, structure your response in two parts:

    Context: {context}
    Question: {question}

    Begin your response with your reasoning process, marking it with <think> tags. Then provide the actual answer.
    For example:
    <think>
    Here I analyze the information...
    These are my considerations...
    This is how I arrived at the answer...
    </think>
    
    Then provide the actual answer to the question.

    Remember: Always wrap your reasoning process in <think> tags, and keep the actual answer separate.
    """
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
    
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Content: {page_content}\nSource: {source}"
    )
    
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt
    )
    
    qa_chain = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # Clean up
    os.remove(temp_file_path)
    
    return qa_chain

# File upload section
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=['pdf'],
    help="Upload a PDF document to analyze"
)

if uploaded_file:
    # Process the uploaded file
    pdf_content = uploaded_file.read()
    if st.session_state.pdf_content != pdf_content:
        st.session_state.pdf_content = pdf_content
        with st.spinner("Processing document..."):
            st.session_state.qa_chain = setup_qa_chain(pdf_content)
        st.success("✅ Document processed successfully!")
    
    # Question input section
    st.markdown("---")
    st.markdown("### Ask a question about your document")
    
    # Create two columns for the question input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question",
            placeholder="e.g., What are the main topics discussed in the document?"
        )
    
    with col2:
        ask_button = st.button("Ask", use_container_width=True)
    
    if question and ask_button:
        with st.spinner("Generating answer..."):
            try:
                # Get answer from the QA chain
                response = st.session_state.qa_chain(question)
                response_text = response["result"]

                # Split into reasoning and answer parts
                if "<think>" in response_text and "</think>" in response_text:
                    # Extract content between think tags
                    think_start = response_text.find("<think>")
                    think_end = response_text.find("</think>") + len("</think>")
                    
                    # Get the reasoning part (between think tags)
                    reasoning_part = response_text[think_start + len("<think>"):think_end - len("</think>")].strip()
                    
                    # Get the answer part (everything after </think>)
                    answer_part = response_text[think_end:].strip()
                else:
                    # If no think tags, consider everything as answer
                    reasoning_part = None
                    answer_part = response_text.strip()

                # Display the response
                st.markdown("### Response")
                
                # Display reasoning if available
                if reasoning_part:
                    st.markdown("#### 🤔 Reasoning Process")
                    st.markdown(f'<div class="reasoning-box">{reasoning_part}</div>', unsafe_allow_html=True)
                
                # Display answer
                st.markdown("#### 💡 Answer")
                st.markdown(f'<div class="answer-box">{answer_part}</div>', unsafe_allow_html=True)
                
                # Show sources
                with st.expander("View Sources"):
                    for i, doc in enumerate(response["source_documents"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
            
else:
    st.info("👆 Please upload a PDF document to get started") 