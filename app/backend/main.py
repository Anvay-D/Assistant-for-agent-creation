from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag.rag_retiever import retrieve_context
from llm.openRouter import call_llm

app = FastAPI(
    title="Qdrant RAG Chat API",
    description="Ask questions over Qdrant documentation using RAG + OpenRouter",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    context_used: bool


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    context = retrieve_context(req.question)

    prompt = f"""
You are answering questions about the Qdrant API.

Use ONLY the information below.
If the answer is not present, say "Not found in documentation".

=====================
CONTEXT:
{context}
=====================

QUESTION:
{req.question}

Answer clearly with steps and examples if applicable.
"""

    answer = call_llm(prompt)

    return ChatResponse(
        answer=answer,
        context_used=bool(context.strip()),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
