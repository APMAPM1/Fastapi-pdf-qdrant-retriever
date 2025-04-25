import os
import logging
import io
import uuid
from typing import List, Dict, Any

# Use dotenv to load environment variables from .env file
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import fitz # PyMuPDF
import nltk

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Qdrant Cloud Configuration (Read from Environment Variables)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Default collection name, can be overridden by environment variable
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents_cloud")

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", 'all-MiniLM-L6-v2')

# --- Global Variables ---
model: SentenceTransformer | None = None
qdrant_client: QdrantClient | None = None
vector_size: int | None = None

# --- Helper Functions ---

def ensure_nltk_data():
    """Downloads NLTK 'punkt' tokenizer if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        logging.info("NLTK 'punkt' tokenizer found.")
    except LookupError:
        logging.info("NLTK 'punkt' tokenizer not found. Attempting download...")
        try:
            nltk.download('punkt', quiet=True)
            logging.info("'punkt' downloaded successfully.")
        except Exception as download_exc:
            logging.error(f"Failed to download NLTK 'punkt': {download_exc}. Sentence tokenization might fail.")
            # Depending on requirements, you might want to raise an error here

def chunk_text_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using NLTK."""
    try:
        sentences = nltk.sent_tokenize(text)
        # Optional: Filter out very short sentences/artifacts
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    except Exception as e:
        logging.error(f"Error tokenizing text with NLTK: {e}")
        # Basic fallback: split by newline and period, filter empty
        chunks = [chunk.strip() for part in text.split('\n') for chunk in part.split('.') if chunk.strip()]
        return [c for c in chunks if len(c) > 10]


def extract_text_from_pdf(file_content: bytes) -> Dict[int, str]:
    """Extracts text from PDF file content, page by page."""
    pages_text = {}
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            logging.info(f"Opened PDF with {len(doc)} pages.")
            for i, page in enumerate(doc):
                text = page.get_text("text", sort=True) # Get text, try to keep logical reading order
                if text.strip(): # Only add pages with actual text
                    pages_text[i + 1] = text # Use 1-based page numbering
            logging.info(f"Extracted text from {len(pages_text)} pages.")
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF file: {e}")
    return pages_text

def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Ensures the Qdrant collection exists."""
    try:
        client.get_collection(collection_name=collection_name)
        logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        # Check if the exception is because the collection doesn't exist
        # This check might need adjustment based on the exact exception type/message from qdrant_client
        if "not found" in str(e).lower() or "doesn't exist" in str(e).lower() or '404' in str(e):
            logging.info(f"Collection '{collection_name}' not found. Creating.")
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    # Optional: Add payload indexing for faster filtering later
                    # payload_schema={
                    #     "filename": models.PayloadSchemaType.KEYWORD,
                    #     "page": models.PayloadSchemaType.INTEGER
                    # }
                )
                logging.info(f"Collection '{collection_name}' created successfully.")
            except Exception as creation_e:
                logging.error(f"Failed to create collection '{collection_name}': {creation_e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to create Qdrant collection.")
        else:
            # Unexpected error trying to access the collection
            logging.error(f"Unexpected error checking/creating collection '{collection_name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to access Qdrant collection.")

# --- FastAPI Application ---
app = FastAPI(
    title="PDF RAG Retrieval API (Qdrant Cloud)",
    description="Upload PDFs, index their content, and perform semantic search.",
    version="1.0.0"
)

# --- Pydantic Models ---
class UploadResponse(BaseModel):
    filename: str
    total_chunks_added: int
    message: str = "PDF processed and indexed successfully."

class SearchResult(BaseModel):
    id: str | int # UUIDs are strings
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict) # To hold filename, page, etc.

class SearchResponse(BaseModel):
    results: List[SearchResult]

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    global model, qdrant_client, vector_size
    logging.info("Application startup: Initializing resources...")

    # 1. Validate Environment Variables
    if not QDRANT_URL or not QDRANT_API_KEY:
        logging.error("FATAL: QDRANT_URL and QDRANT_API_KEY environment variables must be set.")
        raise RuntimeError("Qdrant URL and API Key are required.")

    # 2. Ensure NLTK data is available
    ensure_nltk_data()

    # 3. Initialize Sentence Transformer Model
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        vector_size = model.get_sentence_embedding_dimension()
        if not vector_size:
             raise ValueError("Could not determine vector size from model.")
        logging.info(f"Embedding model loaded. Vector size: {vector_size}")
    except Exception as e:
        logging.error(f"Failed to load Sentence Transformer model: {e}", exc_info=True)
        raise RuntimeError(f"Could not load embedding model {EMBEDDING_MODEL_NAME}") from e

    # 4. Initialize Qdrant Client
    logging.info(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60 # Increase timeout for potentially longer operations
        )
        qdrant_client.get_collections() # Quick check to confirm connection
        logging.info("Qdrant client initialized and connected.")
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant: {e}", exc_info=True)
        raise RuntimeError("Could not connect to Qdrant database.") from e

    # 5. Ensure Qdrant Collection Exists (without deleting)
    if qdrant_client and vector_size:
        logging.info(f"Ensuring Qdrant collection '{COLLECTION_NAME}' exists...")
        setup_qdrant_collection(qdrant_client, COLLECTION_NAME, vector_size)
    else:
        logging.error("Qdrant client or vector size not available. Skipping collection setup.")
        # This state might prevent the app from working, consider raising an error

    logging.info("Application startup complete.")

# --- API Endpoints ---

@app.post("/upload/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF file, extracts text, chunks it, generates embeddings,
    and indexes the chunks in Qdrant.
    """
    global model, qdrant_client
    if not model or not qdrant_client:
        raise HTTPException(status_code=503, detail="Service not ready. Model or DB client not initialized.")
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    logging.info(f"Received file: {file.filename}, content type: {file.content_type}")

    try:
        file_content = await file.read()
        if not file_content:
             raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # 1. Extract text
        logging.info("Extracting text from PDF...")
        pages_text = extract_text_from_pdf(file_content)
        if not pages_text:
             logging.warning(f"No text could be extracted from {file.filename}.")
             # Decide if this is an error or just return success with 0 chunks
             return UploadResponse(filename=file.filename or "Unknown", total_chunks_added=0, message="No text extracted from PDF.")

        # 2. Chunk text and prepare points
        points_to_upsert: List[PointStruct] = []
        all_chunks_texts: List[str] = []
        all_chunks_metadata: List[Dict] = []

        logging.info("Chunking extracted text...")
        for page_num, text in pages_text.items():
            chunks = chunk_text_into_sentences(text)
            for chunk in chunks:
                all_chunks_texts.append(chunk)
                all_chunks_metadata.append({
                    "filename": file.filename or "Unknown",
                    "page": page_num,
                    "original_text": chunk # Store the original chunk text
                })

        if not all_chunks_texts:
            logging.warning(f"No valid text chunks generated from {file.filename}.")
            return UploadResponse(filename=file.filename or "Unknown", total_chunks_added=0, message="No text chunks generated after processing.")

        logging.info(f"Generated {len(all_chunks_texts)} chunks for indexing.")

        # 3. Generate Embeddings (in batches for efficiency)
        logging.info("Generating embeddings...")
        embeddings = model.encode(all_chunks_texts, show_progress_bar=True, batch_size=128) # Adjust batch_size based on your model/memory
        logging.info("Embeddings generated.")

        # 4. Create PointStructs with UUIDs
        for i, (text, vector, metadata) in enumerate(zip(all_chunks_texts, embeddings, all_chunks_metadata)):
             points_to_upsert.append(
                 PointStruct(
                     id=str(uuid.uuid4()), # Generate a unique ID for each chunk
                     vector=vector.tolist(),
                     payload=metadata # Store filename and page number
                 )
             )

        # 5. Upsert to Qdrant (in batches)
        logging.info(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{COLLECTION_NAME}'...")
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert,
            wait=True # Wait for operation to complete
        )
        logging.info("Upsert completed successfully.")

        return UploadResponse(
             filename=file.filename or "Unknown",
             total_chunks_added=len(points_to_upsert)
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        logging.error(f"Error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred while processing the file: {e}")
    finally:
        await file.close() # Ensure the file is closed


@app.get("/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., min_length=3, description="The search query string"),
    limit: int = Query(3, ge=1, le=50, description="Number of results to return")
):
    """Performs similarity search in the Qdrant collection based on the query."""
    global model, qdrant_client
    if not model or not qdrant_client:
        raise HTTPException(status_code=503, detail="Service not ready. Model or DB client not initialized.")
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty.")

    logging.info(f"Received search query: '{q}', limit: {limit}")
    try:
        query_vector = model.encode(q).tolist()
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            with_payload=True # Ensure payload (metadata) is returned
        )
        logging.info(f"Found {len(search_result)} results from Qdrant.")

        results_list = []
        for hit in search_result:
            metadata = hit.payload if hit.payload else {}
            # Ensure 'original_text' exists in payload, otherwise default to N/A
            text_content = metadata.get("original_text", "N/A")
            results_list.append(
                SearchResult(
                    id=hit.id,
                    score=hit.score,
                    text=text_content, # Use the stored original text
                    metadata=metadata # Include all metadata
                )
            )

        return SearchResponse(results=results_list)

    except Exception as e:
        logging.error(f"Error during search for query '{q}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during the search.")


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    if not model:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    if not qdrant_client:
         raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    try:
        # Check Qdrant connection
        qdrant_client.get_collections()
        return {"status": "ok", "message": "Service is running and connected to Qdrant"}
    except Exception as e:
        logging.error(f"Health check failed: Qdrant connection issue: {e}")
        raise HTTPException(status_code=503, detail="Qdrant connection error")


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001)) # Use a different default port if needed
    logging.info(f"Starting Uvicorn server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)