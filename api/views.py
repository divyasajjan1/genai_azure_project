import os
import fitz  # PyMuPDF
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from google import genai
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .models import QueryLog

# Constants - use /home/site/wwwroot/ for persistence on Azure
DB_DIR = "/home/site/wwwroot/chroma_db" 
MODEL_NAME = "gemini-2.0-flash" # Note: check if you meant 1.5 or 2.0
SYSTEM_PROMPT = "You are a PDF Q&A assistant. Answer ONLY using the provided context."

# Initialize Embeddings once to be used by both views
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

class IngestView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response({"error": "No file uploaded"}, status=400)

        file_content = uploaded_file.read()
        doc = fitz.open(stream=file_content, filetype="pdf")
        docs = [Document(page_content=page.get_text(), metadata={"page": i+1, "source": uploaded_file.name}) for i, page in enumerate(doc)]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Use Google Embeddings
        embeddings = get_embeddings()
        
        # Store in Chroma
        db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
        
        return Response({"message": f"Ingested {len(chunks)} chunks successfully from {uploaded_file.name}."})

@method_decorator(csrf_exempt, name='dispatch')    
class AskView(APIView):
    def post(self, request):
        query = request.data.get("question")
        if not query:
            return Response({"error": "Please provide a question"}, status=400)

        # 1. Setup DB & Client - MUST use the same Google embeddings here!
        embeddings = get_embeddings()
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        # Ensure your env variable is GEMINI_API_KEY
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # 2. Retrieve & Generate
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        if not context:
             return Response({"answer": "I couldn't find any relevant information in the uploaded documents.", "sources": []})

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