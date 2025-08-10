import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
LLM_MODEL = "gemini-2.0-flash"
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-small"

llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

rag_prompt = PromptTemplate(
    input_variables=["retrieved_chunks", "user_prompt"],
    template="""
You are a helpful assistant.
Use the context below only if it's relevant.

Context:
{retrieved_chunks}

Question:
{user_prompt}

Answer:
""",
)


def retrieve_context(query: str) -> str:
    try:
        docs = retriever.invoke(query)
        return (
            "\n".join(doc.page_content for doc in docs)
            if docs
            else "No relevant documents found."
        )
    except Exception as e:
        return f"Error retrieving context: {e}"


def build_prompt(user_input: str, context: str) -> str:
    try:
        return rag_prompt.format(retrieved_chunks=context, user_prompt=user_input)
    except Exception as e:
        return f"Error building prompt: {e}"


def generate_response(user_input: str) -> str:
    context = retrieve_context(user_input)
    full_prompt = build_prompt(user_input, context)

    try:
        response = llm.invoke(
            [SystemMessage("You're a quantum AI assistant."), HumanMessage(full_prompt)]
        )
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error generating response: {e}"


@app.post("/query")
def query_rag(request: QueryRequest):
    user_input = request.query
    response = generate_response(user_input)
    return {"response": response}
