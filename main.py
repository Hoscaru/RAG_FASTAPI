from fastapi import FastAPI
from dotenv import load_dotenv

# Initialize FastAPI application
app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Import necesary libraries for LangChain and Cohere
from langchain_cohere import ChatCohere

model = ChatCohere()

# Create a prompt for the model
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto que responde preguntas utilizando información relevante contenida en los documentos proporcionados.

A continuación se presenta una pregunta del usuario, seguida por fragmentos de texto recuperados de una base de datos de documentos.

Tu tarea es generar una respuesta clara, precisa y completa utilizando únicamente la información contenida en los fragmentos recuperados.
No inventes información. Si la respuesta no se encuentra en los fragmentos, responde: “La información no está disponible en los documentos proporcionados.”

Pregunta:
{user_question}

Documentos recuperados:
{retrieved_context}

Respuesta:
""")

# Bind the prompt to the model

rag_model = prompt | model

# Embending model for vectorization
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(model="embed-english-light-v3.0")


# Import necessary libraries for vector store and retriever
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embedding=embeddings)

# Import necessary libraries for document loading, text splitting, vectorization and creating endpoint 
from langchain_community.document_loaders import PyPDFLoader
from fastapi import File, UploadFile
import tempfile
import shutil

@app.post("/load_pdf")
def load_pdf_pdf(file: UploadFile = File(...)):
    """
    Endpoint to load a PDF file and return its content as a list of documents.
    """
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    docs = []
    for doc in documents:
        docs.append(doc.page_content)
    
    # Text splitting the documents into smaller chunks
    from langchain.text_splitter import TokenTextSplitter

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.create_documents(docs)

    # Vectorizing the documents and adding them to the vector store
    
    vector_store.add_documents(split_documents)

    # Return the number of documents loaded
    return {
            "status": "PDF successfully uploaded",
            "pages_loaded": len(documents),
            "chunks_created": len(split_documents)
        }


# Define a Pydantic model for the request prompt
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

# Create a simple endpoint to handle chatbot requests and import necessary libraries for langchain to handle messages types
from langchain.schema import HumanMessage, AIMessage

messages = []

@app.post("/chatbot")
def chatbot_response(chatbot: PromptRequest):
    """
    Endpoint to handle chatbot requests.
    It receives a prompt and returns a response.
    """
    queary = chatbot.prompt
    user_message = HumanMessage(content=queary)
    messages.append(user_message)
    context_docs = vector_store.similarity_search(queary, k=3)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    
    
    response = rag_model.invoke({
        "user_question": queary,
        "retrieved_context": context_text
    })
    messages.append(AIMessage(content=response.content))
    return response.content