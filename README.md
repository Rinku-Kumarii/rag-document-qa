
# RAG Document QA

A Retrieval-Augmented Generation (RAG) based application that allows users to upload PDF documents and ask questions about their content.

## Features

- Upload multiple PDF documents
- Extract text from PDF pages
- Split text into chunks
- Generate embeddings using Sentence Transformers
- Retrieve relevant information using semantic search
- Generate answers using Groq LLM
- Display source filenames and page numbers
- Filter duplicate retrieved chunks

## Technologies Used

- Python
- FastAPI
- Sentence Transformers
- PyTorch
- PyPDF
- Groq API
- Uvicorn

## How It Works

1. Upload PDF documents.
2. Extract text from the documents.
3. Split extracted text into chunks.
4. Convert chunks into vector embeddings.
5. Convert the user's question into an embedding.
6. Retrieve relevant chunks using cosine similarity.
7. Send the retrieved context to the Groq LLM.
8. Return the generated answer with source information.

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd rag-document-qa
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment on Windows

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install fastapi uvicorn pypdf python-multipart sentence-transformers groq python-dotenv
```

### 5. Configure the API key

Create a `.env` file in the project folder:

```text
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own API key. Never upload the `.env` file to GitHub.

### 6. Start the server

```bash
python -m uvicorn main:app --reload
```

Open the API documentation at:

http://127.0.0.1:8000/docs

## Current Limitations

- Uploaded documents are stored in memory and must be uploaded again after restarting the server.
- PDF text extraction depends on the document format.
- Answers depend on the relevance of the retrieved chunks.

## Future Improvements

- Persistent document storage
- Improved chunking and retrieval
- Better support for scanned PDFs
- A user-friendly web interface

## Author

B.Tech Computer Science and Engineering Student
