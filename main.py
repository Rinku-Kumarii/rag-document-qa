
from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
import os
import torch
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="RAG Document QA")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Store chunks and their metadata
stored_chunks = []
stored_page_numbers = []
stored_filenames = []
stored_embeddings = None


@app.get("/")
def home():
    return {
        "message": "RAG Document QA Project is running!"
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global stored_chunks, stored_page_numbers
    global stored_embeddings, stored_filenames

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    try:
        content = await file.read()
        reader = PdfReader(BytesIO(content))

        page_texts = []
        text = ""

        # Extract text page by page
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_texts.append((page_number, page_text))
            text += page_text + "\n"

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in this PDF."
            )

        # Split pages into chunks of 500 characters
        chunk_size = 500
        new_chunks = []
        new_page_numbers = []

        for page_number, page_text in page_texts:
            for i in range(0, len(page_text), chunk_size):
                chunk = page_text[i:i + chunk_size].strip()

                if chunk:
                    new_chunks.append(chunk)
                    new_page_numbers.append(page_number)

        if not new_chunks:
            raise HTTPException(
                status_code=400,
                detail="No text chunks could be created."
            )

        # Generate embeddings for the new PDF
        new_embeddings = model.encode(
            new_chunks,
            convert_to_tensor=True
        )

        # Add new PDF without deleting previously uploaded PDFs
        if stored_embeddings is None:
            stored_chunks = new_chunks
            stored_page_numbers = new_page_numbers
            stored_filenames = [file.filename] * len(new_chunks)
            stored_embeddings = new_embeddings
        else:
            stored_chunks.extend(new_chunks)
            stored_page_numbers.extend(new_page_numbers)
            stored_filenames.extend(
                [file.filename] * len(new_chunks)
            )
            stored_embeddings = torch.cat(
                [stored_embeddings, new_embeddings]
            )

        return {
            "filename": file.filename,
            "total_pages": len(reader.pages),
            "total_characters": len(text),
            "new_chunks": len(new_chunks),
            "total_chunks": len(stored_chunks),
            "total_documents": len(set(stored_filenames)),
            "message": "PDF uploaded and indexed successfully!"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process PDF: {str(e)}"
        )


@app.get("/ask")
def ask_question(question: str):
    if not stored_chunks or stored_embeddings is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:
        # Generate question embedding
        question_embedding = model.encode(
            question,
            convert_to_tensor=True
        )

        # Compare question with all stored chunks
        similarities = util.cos_sim(
            question_embedding,
            stored_embeddings
        )[0]

        # Retrieve the top 3 most relevant chunks
        top_k = min(20, len(stored_chunks))
        top_results = similarities.topk(top_k)

        results = []
        seen_chunks = set()

        for score, index in zip(
            top_results.values.tolist(),
            top_results.indices.tolist()
        ):
            chunk_text = stored_chunks[index].strip()

            # Filter low similarity and duplicate text
            if score >= 0.15 and chunk_text not in seen_chunks:
                seen_chunks.add(chunk_text)

                results.append({
                    "filename": stored_filenames[index],
                    "similarity_score": round(score, 3),
                    "page_number": stored_page_numbers[index],
                    "text": chunk_text
                })

        if not results:
            return {
                "question": question,
                "answer": (
                    "No sufficiently relevant information "
                    "found in the uploaded PDFs."
                ),
                "method": "Semantic search",
                "matching_chunks": 0,
                "sources": []
            }

        # Build context from retrieved chunks
        context = "\n\n".join(
            f"File: {item['filename']}\n"
            f"Page {item['page_number']}:\n{item['text']}"
            for item in results
        )

        # Generate answer using Groq LLM
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the provided document context. "
                        "If the answer is not in the context, say you "
                        "could not find it in the documents. "
                        "Keep the answer clear and concise."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Document context:\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ],
            temperature=0.2
        )

        answer = completion.choices[0].message.content

        return {
            "question": question,
            "answer": answer,
            "matching_chunks": len(results),
            "sources": results,
            "method": "Semantic Search + Groq LLM"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not answer question: {str(e)}"
        )
