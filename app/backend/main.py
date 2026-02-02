from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from router.Router import route_query
from agent_prompt.summarization import SUMMARIZATION_PROMPT
from rag.lang_chain_store import LANGCHAIN_COLLECTION, LANGGRAPH_COLLECTION
from rag.rag_retiever import retrieve_context
from llm.openRouter import call_llm

app = FastAPI(
    title="Qdrant RAG Chat API",
    description="Ask questions over documentation using RAG + LangGraph routing",
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
    # 1️⃣ Decide which KB to use
    decision = route_query(req.question)

    # 2️⃣ Retrieve context
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

    # 3️⃣ Summarize / answer
    prompt = f"""
    You are summarizing technical documentation.

Your goal:
- Compress the content
- Preserve technical accuracy
- Explain concepts clearly
- Do NOT add new information
- You can write python code with examples of API calls and custom API based on the user query with appropriate context and logic.
- You can create an agent based on the user query and the provided context.

Use ONLY the information below.
If the answer is not present, say "Not found in documentation".

=====================
CONTEXT:
{merged_context}
=====================

QUESTION:
{req.question}
"""

    answer = call_llm(prompt, SUMMARIZATION_PROMPT)

    return ChatResponse(
        answer=answer,
        context_used=bool(merged_context.strip()),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
