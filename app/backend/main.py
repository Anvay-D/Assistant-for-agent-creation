from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from router.router import route_query
from agent_prompt.summarization import SUMMARIZATION_PROMPT



from rag.lang_chain_store import (LANGGRAPH_COLLECTION,LANGCHAIN_COLLECTION)
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
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    decision = route_query(req.question)

    contexts = []

    if decision in ("langchain", "both"):
        ctx = retrieve_context(
            req.question,
            collection=LANGCHAIN_COLLECTION,
        )
        if ctx.strip():
            contexts.append(ctx)

    if decision in ("langgraph", "both"):
        ctx = retrieve_context(
            req.question,
            collection=LANGGRAPH_COLLECTION,
        )
        if ctx.strip():
            contexts.append(ctx)

    merged_context = "\n\n".join(contexts)

    prompt = f"""
Use ONLY the information below. If the answer is not present, say "Not found in documentation".

=====================
CONTEXT:
{merged_context}
=====================

QUESTION:
{req.question}
"""

    answer = call_llm(prompt,SUMMARIZATION_PROMPT)

    return ChatResponse(
        answer=answer,
        context_used=bool(merged_context.strip()),
    )

@app.get("/health")
def health():
    return {"status": "ok"}
