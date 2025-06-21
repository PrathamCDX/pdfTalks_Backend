from qdrant_client import AsyncQdrantClient  
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import fitz  
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv, find_dotenv
import tempfile
import pdfplumber
import tempfile
import os
from fastapi import UploadFile

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


qdrant_api_key_write = os.getenv("QDRANT_API_KEY_WRITE")
qdrant_api_key_read = os.getenv("QDRANT_API_KEY_READ")
qdrant_url=os.getenv("QDRANT_URL")
point_chunk_size = 50 

def hello():
    print("Hello from the controller function!")
    
def get_hf_sentence_transformer(sentences):
  hf_sentence_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  hf_sentence_transformer_embeddings = hf_sentence_transformer_model.encode(sentences)
  return hf_sentence_transformer_embeddings

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_hf_transformer(sentences):
  tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
  model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
  with torch.no_grad():
      model_output = model(**encoded_input)

  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
  sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

  return sentence_embeddings.numpy()

async def generate_embeddings(sentences):
  tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
  model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
  with torch.no_grad():
      model_output = model(**encoded_input)

  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
  sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

  return sentence_embeddings.numpy()

def compare_embeddings(embedding_one, embedding_two):
  result_arr =[]
  for i in range(embedding_one.shape[0]):
    result=0 
    for j in range(embedding_one.shape[1]):
      result += embedding_one[i][j]* embedding_two[i][j]
    result_arr.append(result)
  return result_arr

# qdrant connection





# qdrant connection

async def ensure_collection_exists(
    collection_name: str,
    vector_size: int = 384,
    distance: Distance = Distance.DOT,
) -> AsyncQdrantClient:  
    client = AsyncQdrantClient( 
        url=qdrant_url,
        api_key=qdrant_api_key_write,
    )

    # Use async methods
    existing_collections = await client.get_collections()
    collection_names = [col.name for col in existing_collections.collections]

    if collection_name in collection_names:
        print(f"Collection '{collection_name}' already exists.")
    else:
        print(f"Creating collection '{collection_name}'...")
        await client.recreate_collection(  
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print(f"Collection '{collection_name}' created.")

    return client 

async def insert_embeddings(embeddings, sentences, qdrant_client, collection_name):
  try:
    async_qdrant_client = await ensure_collection_exists(collection_name)

    point_array = []

    for i in range(embeddings.shape[0]):
      point_array.append(PointStruct(id=i, vector=embeddings[i], payload= {"sentence" : sentences[i]}))
      
    print("point_array size : ", len(point_array))
    # return 0 
    retry_point_array = []
    for i in range(0, len(point_array), point_chunk_size):
      chunk = point_array[i:i + point_chunk_size]
      try:
        qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=chunk,
        )
        print(f"Inserted chunk starting at {i} with size {len(chunk)}")
      except Exception as e:
        retry_point_array.append(chunk)
        print(f"Error inserting chunk starting at {i}: {e}")
        
        # retry setup 
      for i in range(0, len(retry_point_array), int(point_chunk_size/10)):
        retry_chunk = retry_point_array[i:i + int(point_chunk_size/10)]
        try:
          qdrant_client.upsert(
              collection_name=collection_name,
              wait=True,
              points=retry_chunk,
          )
          print(f"Retried chunk starting at {i} with size {len(retry_chunk)}")
        except Exception as e:
          print(f"Error retrying retry_chunk starting at {i}: {e}")
          
  except Exception as e:
    print(f"Error in insert_embeddings: {e}")
    return {"status": "error", "message": str(e)}

def search_embedding(embeddings,qdrant_client,collection_name, maxLimit=1):
  search_result = qdrant_client.query_points(
      collection_name=collection_name,
      query=embeddings,
      with_payload=True,
      limit=maxLimit
  ).points
  return search_result

def delete_collection(collection_name, qdrant_client):
  qdrant_client.delete_collection(collection_name=collection_name)
  
def get_sentence_array(article):
  # article_sentences: list
  article_sentences = ((sent_tokenize(article)))
  return article_sentences

def get_context ( question, qdrant_client,collection_name, limit=3):
  question_array = []
  question_array.append(question)
  print(question_array)
  question_embedding= get_hf_transformer(question_array)
  res= search_embedding(question_embedding[0], qdrant_client, collection_name, limit)

  context= ""
  for i in range(len(res)):
    context += res[i].payload['sentence'] + " "

  return context

def llm_response( question, context):
  model_name = "deepset/tinyroberta-squad2"

  nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
  QA_input = {
      'question': question,
      'context': context
  }
  res = nlp(QA_input)

  return res

async def convert_pdf_to_text_copilot(pdf_file):
    try:
        with pdfplumber.open(pdf_file.file) as pdf:
            text =  ".".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text.replace("\n", " ") if text else "No text found in PDF"
    except Exception as e:
        print(f"Error extracting text: {e}")
        return "Error extracting text"
 
async def convert_pdf_to_text_large(pdf_file: UploadFile):
    temp_path = None
    try:
        pdf_file.file.seek(0)
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            temp_path = tmp.name
            # Write uploaded file contents to temp file
            contents = await pdf_file.read()
            tmp.write(contents)
        
        # Process the temporary file
        with pdfplumber.open(temp_path) as pdf:
            text = " ".join(
                page.extract_text().replace("\n", " ") 
                for page in pdf.pages 
                if page.extract_text()
            )
            
        return text if text else "No text found in PDF"
    
    except Exception as e:
        print(f"Error extracting text: {e}")
        return f"Error extracting text: {str(e)}"
    
    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        # Reset the file pointer if needed
        await pdf_file.seek(0)
        
        
        

from supabase import create_client, Client
def upload_pdf_to_supabase(
    supabase: Client,
    file, 
    bucket_name: str = "pdfs", 
    file_name: str = None,
    content_type: str = "application/pdf"
):
    
    try:
        if not file_name:
            raise ValueError("file_name parameter is required when uploading a file object")
        
        # Read the file data
        file_data = file.read()
        
        # Upload to Supabase Storage
        res = supabase.storage.from_(bucket_name).upload(
            path=file_name,
            file=file_data,
            file_options={"content-type": content_type}
        )
        
        print(f"Successfully uploaded {file_name} to {bucket_name} bucket")
        
        # Return public URL (requires proper bucket policies)
        return supabase.storage.from_(bucket_name).get_public_url(file_name)
    
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None
      
def trim_pdf_extension(filename):
    if filename.lower().endswith('.pdf'):
        return filename[:-4]
    return filename
  
def chunk_text_with_overlap(text, chunk_size=500, overlap_size=250):
    
    chunks = []
    start = 0
    end = chunk_size
    
    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move the window forward (chunk_size - overlap_size)
        start += (chunk_size - overlap_size)
        end = start + chunk_size
    
    return chunks
