from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time

# Ensure directories exist for storing uploaded PDFs
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('jj'):
    os.mkdir('jj')

# Initialize session state variables
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with user questions. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

if 'vectorstore' not in st.session_state:
    # Initialize vectorstore with OllamaEmbeddings
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="mistral")
    st.session_state.vectorstore = Chroma(
        persist_directory="./jj",
        embedding_function=embeddings
    )

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

if 'chat_history' not in st.session_state:  # Initialize chat history
    st.session_state.chat_history = []


# Streamlit app
st.title("PDF Chatbot")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

if uploaded_file is not None:
    if uploaded_file.name not in st.session_state.uploaded_files:
        # Save the file and process it
        bytes_data = uploaded_file.read()
        file_path = f"files/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        # Load and split the document
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
        all_splits = text_splitter.split_documents(data)

        # Now the vectorstore is initialized and we can add documents
        st.session_state.vectorstore.add_documents(documents=all_splits)

        # Persist the vector store
        st.session_state.vectorstore.persist()

        # Add the uploaded file to the list of uploaded files
        st.session_state.uploaded_files.append(uploaded_file.name)


        # Set up retriever
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()

        # Set up QA chain
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

# Chat UI
st.subheader("Uploaded PDF Files:")
if st.session_state.uploaded_files:
    for file_name in st.session_state.uploaded_files:
        st.write(file_name)
else:
    st.write("No PDF files uploaded yet.")

if st.session_state.qa_chain:
    if user_input := st.chat_input("Ask a question about the PDF..."):
        # Add user input to chat history
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)

        # Generate assistant response
        response = st.session_state.qa_chain({"query": user_input})
        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)

# Display chat history (only once)
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])
