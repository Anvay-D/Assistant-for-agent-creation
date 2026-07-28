# System Design Document: Agentic AI RAG Helper

## 1. Project Overview

Agentic AI RAG Helper is a production-ready Retrieval-Augmented Generation (RAG) chatbot built with LangGraph, LangChain, and Qdrant. The system enables developers to query LangChain, LangGraph, and Qdrant documentation with zero hallucinations by grounding all responses in verified documentation.

**Key Technologies:**
- Backend: FastAPI (Python)
- Frontend: Streamlit
- Vector Database: Qdrant
- LLM Orchestration: LangGraph, LangChain
- LLM Provider: OpenRouter (supports multiple models)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│                    Streamlit Chat UI                             │
│                    (Port: 8600)                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTP POST /chat
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                           │
│                    FastAPI Backend                               │
│                    (Port: 8000)                                  │
│  • CORS Middleware                                               │
│  • Request/Response Models                                       │
│  • Session Management                                            │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐    ┌───────────────────┐    ┌────────────────┐
│   Router      │    │   RAG Layer       │    │   Memory       │
│   Layer       │    │                   │    │   Layer        │
│               │    │ • Query Router    │    │                │
│ • Intent      │    │ • Vector Search   │    │ • Session IDs  │
│   Classification│  │ • Context Retrieval│   │ • Conversation │
│ • LangGraph   │    │ • Document Store  │    │   History      │
│   Workflow    │    │                   │    │                │
└───────────────┘    └───────────────────┘    └────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      LLM Layer        │
                    │   OpenRouter API      │
                    │   • Model Routing     │
                    │   • Response Gen      │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Vector Database     │
                    │      Qdrant           │
                    │   (Port: 6333)        │
                    │                       │
                    │ Collections:          │
                    │ • langchain_docs      │
                    │ • langgraph_docs      │
                    │ • qdrant_api_complete │
                    └───────────────────────┘
```

---

## 3. Directory Structure

```
/root/Assistant-for-agent-creation/
├── app/
│   ├── backend/
│   │   ├── agent_prompt/          # System prompts for different agents
│   │   │   ├── Langchain.py       # LangChain-specific prompt
│   │   │   ├── LangGraph.py       # LangGraph-specific prompt
│   │   │   ├── RouterQuery.py     # Query routing prompt
│   │   │   └── summarization.py   # Response summarization prompt
│   │   ├── llm/
│   │   │   └── openRouter.py      # OpenRouter LLM client
│   │   ├── memory/
│   │   │   └── short_memory.py    # Session-based conversation memory
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── collection_store.py    # Collection name constants
│   │   │   ├── lang_chain_store.py    # LangChain/LangGraph ingestion
│   │   │   ├── qdrant_store.py        # Qdrant API ingestion
│   │   │   └── rag_retiever.py        # Vector search & retrieval
│   │   ├── router/
│   │   │   └── Router.py              # LangGraph-based query router
│   │   ├── main.py                      # FastAPI application entry
│   │   └── requirements.txt             # Backend dependencies
│   └── frontend/
│       ├── chat_ui.py                   # Streamlit chat interface
│       └── requirements.txt             # Frontend dependencies
├── assets/                               # UI screenshots
├── requirements.txt                      # Root dependencies
├── README.md
└── SYSTEM_DESIGN.md
```

---

## 4. Frontend Architecture

### 4.1 Streamlit Chat UI (chat_ui.py)

**Components:**
- **Session State Management**: Maintains chat history in `st.session_state.messages`
- **Chat Interface**: Uses `st.chat_message()` and `st.chat_input()` for native chat UI
- **API Communication**: HTTP client via `requests` library to backend `/chat` endpoint

**Data Flow:**
1. User inputs question via `st.chat_input()`
2. Message appended to session state
3. POST request sent to `http://localhost:8000/chat`
4. Response rendered with `st.markdown()`
5. Assistant response stored in session state

**Configuration:**
- Page title: "Langchain RAG Chat"
- Layout: Centered
- Port: 8600 (default Streamlit port overridden)

---

## 5. Backend Architecture

### 5.1 FastAPI Application (main.py)

**Application Setup:**
```python
app = FastAPI(
    title="Technical Helper RAG Chat API",
    description="Ask questions over documentation using RAG + LangGraph routing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
```

**Middleware:**
- CORS middleware with `allow_origins=["*"]` for cross-origin requests (configurable via settings)

**Configuration Management:**
- Uses Pydantic Settings (`config.py`) for environment-based configuration
- Supports `.env` file loading
- Validates `OPENROUTER_API_KEY` on startup
- Configurable memory turns, log level, and API version

**API Endpoints:**

| Endpoint | Method | Description | Request Model | Response Model |
|----------|--------|-------------|---------------|----------------|
| `/api/v1/chat` | POST | Main chat endpoint | ChatRequest | ChatResponse |
| `/api/v1/health` | GET | Health check | - | {"status": "ok", "version": "v1"} |
| `/docs` | GET | Interactive Swagger UI | - | - |
| `/redoc` | GET | Alternative API docs | - | - |

**API Versioning:**
- All endpoints under `/api/{version}/` prefix
- Default version: `v1` (configurable via `API_VERSION` setting)

**Request/Response Models with Validation:**
```python
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = None

    @field_validator("question")
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")

class ChatResponse(BaseModel):
    answer: str
    context_used: bool
    session_id: str
```

**Error Handling:**
- Global exception handlers for `HTTPException` and generic exceptions
- Standardized `ErrorResponse` format with:
  - Error details (message, code, field)
  - Timestamp (ISO 8601)
  - Request path
  - Request ID for tracing
- Proper HTTP status codes (400, 500)

### 5.2 Request Processing Pipeline

**Step 1: Session Management**
- `get_session_id()` generates UUID if no session_id provided
- `get_short_memory()` retrieves conversation history for session
- Last 5 conversation turns included in context

**Step 2: Query Routing**
- `route_query()` classifies question intent
- Routes to appropriate knowledge base(s):
  - `"langchain"` → LangChain documentation
  - `"langgraph"` → LangGraph documentation
  - `"qdrant"` → Qdrant API documentation
  - `"All"` → Multiple knowledge bases

**Step 3: Context Retrieval**
- Parallel retrieval from selected collections
- Uses `retrieve_context()` from RAG layer
- Merges contexts with `\n\n` separator

**Step 4: LLM Invocation**
- Constructs prompt with merged context + memory
- Calls `call_llm()` with summarization system prompt
- Response added to session memory

**Step 5: Response Construction**
- Returns answer, context usage flag, and session_id
- All operations logged with request ID for tracing
- Exceptions caught and converted to standardized error responses

### 5.2 Logging Infrastructure

**Logging Configuration:**
- Uses Python's `logging` module with configurable level
- Format: `%(asctime)s | %(name)s | %(levelname)s | %(message)s`
- Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL (via `LOG_LEVEL` setting)

**Logged Events:**
- Request processing with truncated question preview
- Router decisions
- Context retrieval (collection name, length)
- Memory operations (turn count)
- LLM generation (answer length)
- Errors with full traceback

**Request Tracing:**
- Each request gets a unique 8-character ID
- Supports `X-Request-ID` header for external tracing
- ID included in all log entries and error responses

---

## 6. RAG Layer Architecture

### 6.1 Collection Management (collection_store.py)

**Collection Names:**
```python
QDRANT_COLLECTION = "qdrant_api_complete"
LANGCHAIN_COLLECTION = "langchain_docs"
LANGGRAPH_COLLECTION = "langgraph_docs"
```

### 6.2 Data Ingestion

#### 6.2.1 LangChain/LangGraph Store (lang_chain_store.py)

**Data Source:** `https://docs.langchain.com/llms.txt`

**Process:**
1. Fetch `llms.txt` containing markdown links to documentation
2. Extract URLs using regex: `r"\((https?://[^)]+)\)"`
3. Filter URLs by keyword ("langchain" or "langgraph")
4. Fetch full document content for each URL
5. Create LangChain `Document` objects with metadata
6. Split documents using `RecursiveCharacterTextSplitter`:
   - chunk_size: 1200
   - chunk_overlap: 150
7. Store in Qdrant with embeddings

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

#### 6.2.2 Qdrant API Store (qdrant_store.py)

**Data Source:** Qdrant OpenAPI specs from GitHub

**OpenAPI Files Ingested:**
- openapi-collections.ytt.yaml
- openapi-points.ytt.yaml
- openapi-main.ytt.yaml
- openapi-shards.ytt.yaml
- openapi-shard-snapshots.ytt.yaml
- openapi-snapshots.ytt.yaml
- openapi-service.ytt.yaml
- openapi-cluster.ytt.yaml

**Process:**
1. Download YAML specs from GitHub raw URLs
2. Parse OpenAPI paths and methods
3. Create `Document` objects with:
   - Endpoint details (method, path, operationId)
   - Request/response schemas
   - Metadata (source, deprecated status)
4. Split with `RecursiveCharacterTextSplitter`:
   - chunk_size: 800
   - chunk_overlap: 120
5. Store in Qdrant collection

### 6.3 Context Retrieval (rag_retiever.py)

**Components:**
- `QdrantClient`: Connected to `http://localhost:6333`
- `SentenceTransformer`: Embedding model for query encoding

**Retrieval Process:**
1. Encode query to vector using `all-MiniLM-L6-v2`
2. Execute `client.query_points()` with:
   - collection_name: targeted collection
   - query: encoded vector
   - limit: top_k (default 5)
   - with_payload: True
3. Extract payloads and format results:
   ```
   [Source: {source} | Score: {score:.3f}]
   {text[:800]}...
   ```
4. Join multiple contexts with `\n\n---\n\n`

**Error Handling:**
- Empty query check
- Fallback for missing payload fields
- Graceful handling of missing scores

---

## 7. Router Layer Architecture

### 7.1 LangGraph-Based Router (Router.py)

**State Definition:**
```python
class RouterState(TypedDict):
    input: str
    decision: Literal["qdrant", "langchain", "langgraph", "All"]
```

**Router Graph Structure:**
```
START → router_node → END
```

**Router Node Logic:**
1. Receives input question in state
2. Calls LLM with `ROUTER_PROMPT` system prompt
3. Parses LLM response to extract decision
4. Returns decision in lowercase

**Routing Prompt Rules:**
- "langchain" → chains, LCEL, retrievers, vectorstores, agents
- "langgraph" → graphs, nodes, edges, state, routing, Command
- "All" → questions involving BOTH LangChain and LangGraph
- "qdrant" → Qdrant, vector databases, API calls

---

## 8. LLM Layer Architecture

### 8.1 OpenRouter Client (openRouter.py)

**Configuration (config.py required):**
```python
OPENROUTER_API_KEY = "your_api_key"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

**LLM Call Process:**
1. POST request to OpenRouter API
2. Headers: Bearer auth + JSON content type
3. Payload:
   - model: configured model name
   - messages: system prompt + user prompt
   - temperature: 0.2 (deterministic)
4. Timeout: 60 seconds
5. Extracts `choices[0].message.content` from response

**Error Handling:** `response.raise_for_status()` for HTTP errors

---

## 9. Memory Layer Architecture

### 9.1 Short-Term Memory (short_memory.py)

**Storage:** In-memory Python dictionary
```python
memory_store = {}  # session_id -> list of conversation turns
```

**Functions:**

| Function | Purpose | Return Value |
|----------|---------|--------------|
| `get_session_id(session_id)` | Validate or generate session ID | Valid session_id (UUID) |
| `get_short_memory(session_id)` | Retrieve conversation history | List of message dicts |
| `add_to_memory(session_id, user, assistant)` | Append conversation turn | None (mutates store) |

**Conversation Turn Format:**
```python
{
    "user": "user question",
    "assistant": "assistant response"
}
```

**Usage in Main Pipeline:**
- Retrieves last 5 turns: `memory[-5:]`
- Formats as: `"User: {...}\nAssistant: {...}\n"`
- Includes in LLM prompt for context continuity

### 9.2 Frontend Session Implementation

**Implementation (chat_ui.py:14-41):**
The frontend now properly implements session management to enable short-term memory:

1. **Session Initialization** (lines 14-15):
   ```python
   if "session_id" not in st.session_state:
       st.session_state.session_id = str(uuid.uuid4())
   ```

2. **Request with Session** (lines 32-35):
   ```python
   payload = {
       "question": prompt,
       "session_id": st.session_state.session_id
   }
   res = requests.post(API_URL, json=payload)
   ```

3. **Session Sync** (lines 40-41):
   ```python
   if "session_id" in data:
       st.session_state.session_id = data["session_id"]
   ```

**Result:** The short-term memory system is now fully functional. Each user session maintains conversation context across messages, with the backend including the last 5 turns in LLM prompts for context-aware responses.

---

## 10. System Prompts

### 10.1 Router Prompt (RouterQuery.py)
Classifies query intent to determine knowledge base routing.

### 10.2 Summarization Prompt (summarization.py)
Instructs LLM to:
- Compress content while preserving accuracy
- Explain concepts clearly
- NOT add new information
- Generate Python code/API examples from context
- Create agents from provided context

### 10.3 Domain-Specific Prompts
- **Langchain.py**: LangChain API documentation specialist
- **LangGraph.py**: LangGraph API documentation specialist
- Both enforce: answer only from context, no deprecated APIs

---

## 11. Data Flow Diagrams

### 11.1 Query Processing Flow

```
User Query
    │
    ▼
Frontend (Streamlit)
    │ POST /chat
    ▼
FastAPI (main.py)
    │
    ├─→ get_session_id() ──→ Memory Layer
    │                          │
    │                          └─→ get_short_memory()
    │                               (last 5 turns)
    │
    ├─→ route_query() ──→ Router Layer (LangGraph)
    │                      │
    │                      └─→ LLM Classification
    │                           (langchain/langgraph/qdrant/All)
    │
    ├─→ retrieve_context() × N
    │      │
    │      └─→ RAG Layer
    │           │
    │           ├─→ Encode query (SentenceTransformer)
    │           ├─→ Qdrant vector search
    │           └─→ Format results with scores
    │
    ├─→ Merge contexts
    │
    ├─→ Construct LLM prompt
    │    (context + memory + question)
    │
    ├─→ call_llm()
    │      │
    │      └─→ OpenRouter API
    │           │
    │           └─→ GPT-4o-mini (temp=0.2)
    │
    ├─→ add_to_memory()
    │
    └─→ Return ChatResponse
         │
         ▼
Frontend renders answer
```

### 11.2 Data Ingestion Flow

```
Ingestion Scripts
    │
    ├─→ lang_chain_store.py
    │    │
    │    ├─→ Fetch llms.txt
    │    ├─→ Extract & filter URLs
    │    ├─→ Fetch documentation pages
    │    ├─→ Create Document objects
    │    ├─→ Chunk documents (1200/150)
    │    └─→ Store in Qdrant
    │         (langchain_docs, langgraph_docs)
    │
    └─→ qdrant_store.py
         │
         ├─→ Fetch OpenAPI YAML files
         ├─→ Parse endpoints/methods
         ├─→ Create Document objects
         ├─→ Chunk documents (800/120)
         └─→ Store in Qdrant
              (qdrant_api_complete)
```

---

## 12. Configuration Requirements

### 12.1 Backend Configuration (config.py)

Must be created in `app/backend/`:
```python
OPENROUTER_API_KEY = "your_api_key"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
QDRANT_URL = "http://localhost:6333"
```

### 12.2 Infrastructure Requirements

**Qdrant Vector Database:**
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_data:/qdrant/storage \
    qdrant/qdrant
```

**Python Environment:**
- Python 3.13+
- All dependencies from requirements.txt installed

---

## 13. Component Interaction Summary

| Component | Responsibility | Key Dependencies |
|-----------|---------------|------------------|
| `chat_ui.py` | User interface, session state | streamlit, requests |
| `main.py` | API orchestration, request pipeline | fastapi, pydantic |
| `Router.py` | Query intent classification | langgraph, openRouter |
| `rag_retiever.py` | Vector similarity search | qdrant-client, sentence-transformers |
| `lang_chain_store.py` | LangChain/LangGraph ingestion | langchain-*, requests, qdrant-client |
| `qdrant_store.py` | Qdrant API ingestion | httpx, pyyaml, langchain-* |
| `openRouter.py` | LLM API client | requests, config |
| `short_memory.py` | Conversation persistence | uuid |

---

## 14. Known Issues & Improvements

**Identified Issues (Fixed):**
1. ~~`short_memory.py` contained unused FastAPI/Pydantic imports~~ ✓ Fixed
2. ~~`get_session_id()` had incorrect logic for new session creation~~ ✓ Fixed

**Potential Improvements:**
1. Add persistent memory storage (Redis/PostgreSQL) for production
2. Implement rate limiting on API endpoints
3. Add authentication/authorization layer
4. Implement structured logging with levels
5. Add request validation and error handling middleware
6. Create unit tests for router and retrieval logic
7. Add metrics/monitoring (Prometheus, Grafana)
8. Implement streaming responses for better UX
9. Add support for multiple embedding models
10. Create Docker Compose for full stack deployment

---

## 15. Development Workflow

**Data Ingestion (One-time Setup):**
```bash
cd app/backend/rag
python3 lang_chain_store.py   # Ingest LangChain/LangGraph docs
python3 qdrant_store.py       # Ingest Qdrant API specs
```

**Running the System:**
```bash
# Terminal 1: Backend
cd app/backend
uvicorn main:app --reload

# Terminal 2: Frontend
cd app/frontend
streamlit run chat_ui.py --server.port 8600
```

**Access Points:**
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Frontend UI: `http://localhost:8600`
- Qdrant Dashboard: `http://localhost:6333/dashboard`

---

## 16. Docker Deployment Architecture

### 16.1 Container Orchestration

The system uses Docker Compose for simplified multi-component deployment:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Frontend   │  │   Backend    │  │   Qdrant     │           │
│  │  Streamlit   │◄─┤   FastAPI    │◄─┤  Vector DB   │           │
│  │   Port 8600  │  │   Port 8000  │  │   Port 6333  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│         │                 │                 │                   │
│         │                 │                 │                   │
│  ┌──────▼─────────────────▼─────────────────▼──────┐           │
│  │              Shared Network (default)            │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  ┌──────────────────────────────┐                               │
│  │      Data Ingestion          │                               │
│  │   (One-time Container)       │                               │
│  └──────────────────────────────┘                               │
│                                                                  │
│  ┌──────────────────────────────┐                               │
│  │      Persistent Volume       │                               │
│  │      qdrant_storage          │                               │
│  └──────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### 16.2 Docker Services

| Service | Purpose | Dependencies | Health Check |
|---------|---------|--------------|--------------|
| `qdrant` | Vector database | None | `/health` endpoint |
| `data-ingestion` | Populate Qdrant collections | qdrant (healthy) | N/A (one-shot) |
| `backend` | FastAPI server | qdrant, data-ingestion | `/api/v1/health` |
| `frontend` | Streamlit UI | backend (healthy) | N/A |

### 16.3 Deployment Commands

```bash
# Full deployment with build
make run

# Stop all services
make stop

# Clean all data
make clean

# Run ingestion only
make ingest
```

### 16.4 Environment Configuration

Create `.env` from template:
```bash
make setup  # Copies .env.example to .env
```

Required variables:
- `OPENROUTER_API_KEY` - LLM provider API key
- `QDRANT_URL` - Auto-set to internal Docker network

### 16.5 Benefits of Docker Deployment

1. **Simplified Setup**: Single command starts entire stack
2. **Dependency Management**: All services versioned in containers
3. **Network Isolation**: Internal Docker network for service communication
4. **Data Persistence**: Qdrant data survives container restarts
5. **Health Monitoring**: Automatic service health checks
6. **Reproducible Environment**: Consistent across all deployments