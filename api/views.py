import os
from rest_framework.views import APIView
from rest_framework.response import Response
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz  # PyMuPDF
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .models import QueryLog

# Create your views here.
# Constants
DB_DIR = "chroma_db"
PDF_PATH = "data/sample.pdf"
MODEL_NAME = "gemini-2.5-flash" 
SYSTEM_PROMPT = "You are a PDF Q&A assistant. Answer ONLY using the provided context."

class IngestView(APIView):
    # Ensure you have these parsers to handle Postman's form-data
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        # 1. Get the file from the request, NOT a hardcoded path
        uploaded_file = request.FILES.get('file')
        
        if not uploaded_file:
            return Response({"error": "No file uploaded"}, status=400)

        # 2. Open the file directly from memory using fitz (PyMuPDF)
        file_content = uploaded_file.read()
        doc = fitz.open(stream=file_content, filetype="pdf")
        # 3. Continue with your RAG logic
        docs = [Document(page_content=page.get_text(), metadata={"page": i+1}) for i, page in enumerate(doc)]

        # 4. Chunk & Embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 5. Store
        db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
        return Response({"message": f"Ingested {len(chunks)} chunks successfully from {uploaded_file.name}."})

@method_decorator(csrf_exempt, name='dispatch')    
class AskView(APIView):
    def post(self, request):
        query = request.data.get("question")
        if not query:
            return Response({"error": "Please provide a question"}, status=400)

        # 1. Setup DB & Client
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # 2. Retrieve & Generate
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
        
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        QueryLog.objects.create(
            question=query,
            answer=response.text,
            sources=[d.metadata for d in docs]
        )
        
        return Response({
            "answer": response.text,
            "sources": [d.metadata for d in docs]
        })