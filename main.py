import traceback
from qdrant_client import QdrantClient
from fastapi import FastAPI
from pydantic import BaseModel
from controllerFunctions import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import os
import uuid
import shutil
from typing import Optional
import uvicorn
import PyPDF2
import io

from dotenv import load_dotenv, find_dotenv
from fastapi.middleware.cors import CORSMiddleware

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
# print(os.getenv("SECRET"))


app = FastAPI()

origins = [
    
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Origins that can access the API
    allow_credentials=True,
    allow_methods=["*"],              # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],              # Authorization, Content-Type, etc.
)

# qdrant connection
qdrant_api_key_write = os.getenv("QDRANT_API_KEY_WRITE")
qdrant_api_key_read = os.getenv("QDRANT_API_KEY_READ")
qdrant_url=os.getenv("QDRANT_URL")

qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_api_key_write,
    timeout=20.0, 
    
)
print("Qdrant client initialized")
print(type(qdrant_client))
print(qdrant_client.get_collections())
print(os.getcwd()) 


class PredictRequest(BaseModel):
    foo: str | None = None
    info : str | None = "N0 one"
    num : int | None = 0
  

@app.post("/")
def read_root():
    return {"message": "Hello, FastAPI!"}



class GetPdfRequest(BaseModel):
    pdfData: str  | None = None


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

        
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), collection_name: Optional[str] = "test_collection_2"):
    

    try:
        print("upload_file")
        context_text =await convert_pdf_to_text_large(file)
        sentences = get_sentence_array(context_text)
        embeddings =await generate_embeddings(sentences)
        op_info =await  insert_embeddings(embeddings, sentences, qdrant_client, collection_name)
        print("embeddings : ")
        print(len(embeddings[0]))
        print(op_info)
        
        return JSONResponse(content={"filename": file.filename, "file": "file"})
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

    
class getanswer(BaseModel):
    question: str | None =" "
    collection_name: str | None = "test_collection"
    limit: int | None = 3
    
@app.post('/getanswer')
async def get_answer(request: getanswer):
    print("getanswer called")
    try:
        print("getans")
        question = request.question
        print(request)
        embeddings = generate_embeddings([question])
        context = get_context(question, qdrant_client, collection_name=request.collection_name, limit=request.limit)
        print("context : ")
        print(context)
        answer = llm_response(question, context)
        print(answer['answer'])
        return JSONResponse(content= {"answer" : answer['answer']}, status_code=200)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)