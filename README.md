# PDF RAG Retrieval API with FastAPI and Qdrant Cloud

This project implements an API for uploading PDF documents, processing their content, and performing semantic searches using FastAPI, Sentence Transformers, and Qdrant Cloud. It serves as a retrieval component suitable for Retrieval-Augmented Generation (RAG) systems working with PDF sources.

## Features

*   **PDF Upload:** `/upload/` endpoint accepts PDF files.
*   **Text Extraction:** Uses `PyMuPDF` (fitz) to efficiently extract text page by page.
*   **Chunking:** Splits extracted text into manageable chunks (currently sentences via NLTK).
*   **Vector Indexing:** Generates embeddings using `sentence-transformers` and stores chunks with metadata (filename, page number) in Qdrant Cloud using unique UUIDs for each chunk.
*   **Semantic Search:** `/search/` endpoint finds relevant text chunks based on query meaning, not just keywords.
*   **Qdrant Cloud Integration:** Leverages Qdrant Cloud for scalable vector storage and search.
*   **Persistent Collection:** Unlike the previous example, the Qdrant collection is **not** deleted on startup, allowing data to accumulate across uploads and restarts.
*   **FastAPI Backend:** Provides a robust and fast web API framework with automatic documentation.
*   **Configuration:** Uses environment variables (`.env` file) for Qdrant credentials and other settings.
*   **Health Check:** Includes a `/health` endpoint.

## Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Git
*   A [Qdrant Cloud](https://cloud.qdrant.io/) account (or a local Qdrant instance). You will need:
    *   Your Qdrant Cluster URL.
    *   A Qdrant API Key.
*   `libgl1` or similar (required by OpenCV, which is often a dependency of `PyMuPDF` or `sentence-transformers` on Linux):
    ```bash
    # On Debian/Ubuntu
    sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
    # On CentOS/Fedora
    sudo yum install -y mesa-libGL
    ```

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    # Replace with your repo URL if you put it on GitHub
    git clone https://github.com/APMAPM1/Fastapi-pdf-qdrant-retriever.git
    cd fastapi-pdf-qdrant-retriever
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run the application, NLTK might download the 'punkt' tokenizer data.*

4.  **Configure Environment Variables:**
    Create a file named `.env` in the project's root directory. **Do not commit this file to Git.** Add your Qdrant Cloud credentials:
    ```dotenv
    # .env file
    QDRANT_URL=https://your-qdrant-cluster-url.cloud.qdrant.io:6333
    QDRANT_API_KEY=your_qdrant_api_key_here

    # Optional overrides
    # COLLECTION_NAME=my_custom_pdf_collection
    # EMBEDDING_MODEL_NAME=all-mpnet-base-v2 # Example of using a different model
    # PORT=8001
    ```
    Replace the placeholder URL and API Key with your actual Qdrant Cloud details.

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload --port 8001
