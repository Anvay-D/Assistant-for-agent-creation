# Qdrant RAG Helper

This project provides a **Qdrant RAG-based chatbot** to help coders and programmers query the latest Qdrant documentation, APIs, and code examples without hallucinations using a **Retrieval-Augmented Generation (RAG)** approach.

---

## Features

* Query Qdrant documentation using natural language.
* Retrieves relevant context from Qdrant collection to ensure accurate answers.
* Uses OpenRouter LLM for answer generation.
* Streamlit-based frontend for interactive chat interface.

---

## Requirements

* Python 3.13+
* Qdrant running on `localhost:6333`
* Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## Configurations
* Set the parameters OPENROUTER_API_KEY , OPENROUTER_URL , OPENROUTER_MODEL , QDRANT_URL in app/backend/config.py
* LLM can be used based on choice like any model and provider OPENROUTER

---

## Starting the Backend

Navigate to the root of the project and run:

```bash
uvicorn app.backend.main:app --reload
```

* **Backend URL:** `http://localhost:8000`
* Provides `/chat` endpoint for frontend to query.
* Health check endpoint available at `/health`.

---

## Starting the Frontend

Run the Streamlit frontend:

```bash
streamlit run app.frontend.chat_ui.py --server.port 8600
```

* **Frontend URL:** `http://localhost:8600`
* Chat interface to type questions and get answers.
* Displays source documents from Qdrant used to generate answers.

---

## How it works

1. **Frontend**: User enters a question in the chat UI.
2. **Backend**: Receives the question and queries Qdrant vector database for relevant documents.
3. **LLM**: Backend sends the question and retrieved context to OpenRouter LLM.
4. **Response**: Backend returns the answer to frontend.
5. **Frontend**: Displays answer with source information for transparency.

This ensures answers are **grounded in actual documentation** and reduces hallucination.

---

## Usage Example

* Ask questions like:

  * "How can I search a point in Qdrant?"
  * "Show me how to use Qdrant batch insert API."
* The system will return the answer along with relevant context from the Qdrant docs.

---

## Notes

* Ensure Qdrant is running and the collections are populated.
* Ensure your OpenRouter API key is set in the backend environment.
* Frontend and backend communicate via HTTP; frontend never talks to Qdrant directly.

---

## Help & Support

For issues or help:

* Check that all dependencies are installed.
* Ensure the correct ports are used (`8000` backend, `8600` frontend).
* Verify Qdrant collections exist and are populated with documents.
* OpenRouter LLM API key should be valid
