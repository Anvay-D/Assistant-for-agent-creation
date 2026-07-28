"""FastAPI application entry point with proper error handling and validation."""

import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from router.Router import route_query
from agent_prompt.summarization import SUMMARIZATION_PROMPT
from rag.lang_chain_store import LANGCHAIN_COLLECTION, LANGGRAPH_COLLECTION
from rag.qdrant_store import QDRANT_COLLECTION
from rag.rag_retiever import retrieve_context
from llm.openRouter import call_llm
from memory.short_memory import get_session_id, get_short_memory, add_to_memory
from config import settings
from models.error import ErrorResponse, ErrorDetail

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Technical Helper RAG Chat API",
    description="Ask questions over documentation using RAG + LangGraph routing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (allow all origins as per requirement)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="User question about documentation",
        json_schema_extra={"example": "How do I create an agent in LangGraph?"}
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for conversation continuity",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"}
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="LLM-generated answer")
    context_used: bool = Field(..., description="Whether context from RAG was used")
    session_id: str = Field(..., description="Session identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To create an agent in LangGraph...",
                "context_used": True,
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with standardized error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc.detail),
                code=f"HTTP_{exc.status_code}",
            ),
            path=str(request.url.path),
            request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())[:8]),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                message="An unexpected error occurred",
                code="INTERNAL_ERROR",
            ),
            path=str(request.url.path),
            request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())[:8]),
        ).model_dump(),
    )


@app.post(
    f"/api/{settings.API_VERSION}/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Process a chat query",
    description="Routes query to appropriate knowledge base(s), retrieves context, and generates grounded answer",
)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """
    Process a user question and return an answer.

    The endpoint:
    1. Routes the query to determine relevant knowledge bases
    2. Retrieves context from selected vector stores
    3. Includes recent conversation history for context
    4. Generates an answer using the configured LLM
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    logger.info(f"[{request_id}] Processing question: {req.question[:50]}...")

    try:
        # Get or create session
        session_id = get_session_id(req.session_id)
        logger.debug(f"[{request_id}] Using session: {session_id}")

        # Retrieve conversation memory
        memory = get_short_memory(session_id)
        memory_text = "\n".join(
            f"User: {m['user']}\nAssistant: {m['assistant']}"
            for m in memory[-settings.MAX_MEMORY_TURNS :]
        )
        logger.debug(f"[{request_id}] Retrieved {len(memory)} memory turns")

        # Route query to appropriate knowledge base(s)
        decision = route_query(req.question)
        logger.info(f"[{request_id}] Router decision: {decision}")

        # Retrieve context from selected collections
        contexts: list[str] = []
        collection_map = {
            "langchain": LANGCHAIN_COLLECTION,
            "langgraph": LANGGRAPH_COLLECTION,
            "qdrant": QDRANT_COLLECTION,
        }

        for kb in (decision, "All"):
            if kb in collection_map:
                ctx = retrieve_context(req.question, collection=collection_map[kb])
                if ctx.strip() and ctx != "No relevant context found.":
                    contexts.append(ctx)
                    logger.debug(f"[{request_id}] Retrieved context from {kb}")

        merged_context = "\n\n".join(contexts)
        logger.info(f"[{request_id}] Total context length: {len(merged_context)} chars")

        # Construct LLM prompt
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
Previous conversation history:
{memory_text}

QUESTION:
{req.question}
""".strip()

        # Call LLM
        answer = call_llm(prompt, SUMMARIZATION_PROMPT)
        logger.info(f"[{request_id}] Generated answer length: {len(answer)} chars")

        # Store in memory
        add_to_memory(session_id, req.question, answer)

        return ChatResponse(
            answer=answer,
            context_used=bool(merged_context.strip()),
            session_id=session_id,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}",
        )


@app.get(
    f"/api/{settings.API_VERSION}/health",
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Returns the health status of the API",
)
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "version": settings.API_VERSION}
