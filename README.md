# RAG API

This project implements a Retrieval-Augmented Generation (RAG) API using FastAPI. It combines document retrieval with a language model to provide context-aware responses to user queries.

## Features

- Extracts text from a PDF file.
- Splits the text into manageable chunks for processing.
- Creates a vector store using embeddings for efficient document retrieval.
- Uses a language model to generate responses based on retrieved context.
- Exposes an API endpoint for querying the system.

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dshathwikr/rag-app.git
   cd rag-app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the `.env` file with your API keys:
   - Example:
     ```
     GOOGLE_API_KEY=your-google-api-key
     ```

## Usage

1. **Prepare the PDF**: Place the PDF file to be processed in the project directory and update the `PDF_PATH` in `vectorizer.py` if necessary.

2. **Generate the Vector Store**:
   Run the following command to extract text, split it into chunks, and create the vector store:
   ```bash
   python vectorizer.py
   ```

3. **Start the API Server**:
   Launch the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

4. **Query the API**:
   Use a tool like `curl` or Postman to send a POST request to the `/query` endpoint:
   ```bash
   curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query": "Your question here"}'
   ```

## Project Structure

- `vectorizer.py`: Handles text extraction, chunking, and vector store creation.
- `main.py`: Implements the FastAPI server and query handling.
- `requirements.txt`: Lists the required Python packages.
- `.env`: Stores environment variables like API keys.

## Notes

- Ensure the `GOOGLE_API_KEY` in the `.env` file is valid for the Google Generative AI integration.
- The vector store is saved in the `chroma_db` directory.