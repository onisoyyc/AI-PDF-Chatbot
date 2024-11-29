# AI-PDF-Chatbot
# PDF Chatbot with LangChain, Mistral AI, and Streamlit  

This project is inspired by the article ["Building a Local PDF Chat Application with Mistral 7B LLM, LangChain, Ollama, and Streamlit"](https://medium.com/@harjot802/building-a-local-pdf-chat-application-with-mistral-7b-llm-langchain-ollama-and-streamlit-67b314fbab57) by Harjot. While the original implementation used Ollama, we’ve modified it to explicitly run the Mistral AI model using Docker for enhanced flexibility and compatibility.  

## Features  
- Upload PDF documents and interact with them using a chatbot.  
- Leverages LangChain for conversational AI and Mistral 7B LLM for natural language processing.  
- Conversation history and contextual memory for a more engaging experience.  

## Requirements  
- Python 3.9 or newer.  
- [pipx](https://pipxproject.github.io/pipx/) for isolated installations of Python packages.  
- [Docker](https://www.docker.com/) for running the Mistral AI model.
  Ensure you have Docker installed on your system. If you don’t have Docker, you can follow the [official Docker installation guide](https://docs.docker.com/get-docker/).  
  (https://docs.docker.com/guides/rag-ollama/?uuid=A9A7BFE4-7C9E-40FB-8218-92AFCD197BB9](https://docs.docker.com/guides/rag-ollama/?uuid=A9A7BFE4-7C9E-40FB-8218-92AFCD197BB9)

## Installation  

Follow these steps to set up the project.  

### 1. Clone the Repository  
```bash  
git clone https://github.com/onisoyyc/pdf-chatbot.git  
cd pdf-chatbot  
```  

### 2. Install Dependencies  
This project uses `pipx` for managing dependencies in isolated environments.  

```bash  
pipx install streamlit  
pipx inject streamlit langchain langchain-community chromadb  
pipx inject streamlit pdfplumber pypdf  
```  

### 3. Run the Mistral Model with Docker  
Ensure Docker is installed and running on your system. Use the following command to execute the Mistral model via Docker:  
```bash  
docker exec -it ollama ollama run mistral  
```

### 4. Run the Streamlit App  
Launch the Streamlit application with the following command:  
```bash  
streamlit run app.py  
```  

## How It Works  
1. **Upload PDF**: Upload a PDF document using the interface.  
2. **Process Content**: The PDF content is split into manageable chunks and stored in a vector database for efficient retrieval.  
3. **Ask Questions**: Interact with the chatbot by asking questions related to the uploaded PDF.  

## Changes from the Original Implementation  
I used `pipx` for dependency management to isolate the Python environment for the project. This caused some issues within the app.py python 
code. So I made some slight modifications to that to fix any issues that arose.

Error Handling:
Ensures that the vector store and retriever are initialized only when a PDF is uploaded.

Persistent Vector Store:
Uses persist_directory="jj" to save the vector store for reuse across sessions.
Retriever and QA Chain Initialization:

Ensures that the QA chain is correctly linked to the retriever after a PDF is uploaded.
Chat Input:
Adds user input to the chain and retrieves responses dynamically.


## Dependencies  
The project uses the following libraries:  
- **Streamlit**: For the user interface.  
- **LangChain**: For building the conversational pipeline.  
- **LangChain Community Modules**: Adds support for Ollama models and embeddings.  
- **ChromaDB**: A vector database for fast content retrieval.  
- **PyPDFLoader**: For loading and processing PDF files.  
- **Mistral AI**: A powerful open-source LLM for conversational AI.  

## Project Structure  
```plaintext  
.  
├── app.py          # Main application file  
├── files/          # Folder to store uploaded PDFs  
├── jj/             # Folder for persisted vector store  
└── README.md       # Project documentation  
```  

## Acknowledgments  
This project is heavily inspired by [Harjot's article on Medium](https://medium.com/@harjot802/building-a-local-pdf-chat-application-with-mistral-7b-llm-langchain-ollama-and-streamlit-67b314fbab57).  

## License  
This project is licensed under the MIT License. See the `LICENSE` file for more details.  
```  
