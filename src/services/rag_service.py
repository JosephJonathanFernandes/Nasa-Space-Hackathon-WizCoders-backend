import os
import json
import time
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from src.api.endpoints import ChatRequest, ChatResponse
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

class RAGService:
    def __init__(self):
        self._retriever = None
        self._llm = None

    def get_retriever(self):
        if self._retriever is not None:
            return self._retriever
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try:
            vectordb = Chroma(persist_directory="chroma_db", collection_name="testing", embedding_function=embeddings)
        except TypeError:
            vectordb = Chroma(persist_directory="chroma_db", collection_name="testing", embedding=embeddings)
        self._retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        return self._retriever

    def get_llm(self):
        if self._llm is not None:
            return self._llm
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(model=os.getenv("MODEL_NAME", "openai/gpt-oss-20b"), api_key=groq_key)
        return self._llm

    async def chat(self, req: ChatRequest):
        retriever = self.get_retriever()
        llm = self.get_llm()
        try:
            docs = retriever.get_relevant_documents(req.question)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
        retrieved_texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
        if llm is None:
            answer = "\n\n".join(retrieved_texts)
            answer = f"(No LLM configured) Retrieved passages:\n\n{answer}"
            return ChatResponse(answer=answer, retrieved=retrieved_texts)
        prompt = PromptTemplate(input_variables=["context", "question"], template="""
You are an assistant. Use the retrieved context to answer the question.

Context:
{context}

Question:
{question}
""")
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
        result = chain.run(req.question)
        return ChatResponse(answer=result, retrieved=retrieved_texts)

    async def stream_chat(self, question: str):
        retriever = self.get_retriever()
        llm = self.get_llm()
        try:
            docs = retriever.get_relevant_documents(question)
        except Exception as e:
            def error_gen():
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                yield "event: done\ndata: \n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")
        retrieved_texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
        async def event_generator():
            for txt in retrieved_texts:
                payload = json.dumps({"text": txt})
                yield f"event: context\ndata: {payload}\n\n"
            if llm is None:
                answer = "\n\n".join(retrieved_texts)
                yield f"event: chunk\ndata: {json.dumps({'text': answer})}\n\n"
                yield "event: done\ndata: \n\n"
                return
            prompt = PromptTemplate(input_variables=["context", "question"], template="""
You are an assistant. Use the retrieved context to answer the question.

Context:
{context}

Question:
{question}
""")
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
            result = chain.run(question)
            tokens = result.split()
            buffer = ""
            for i, tk in enumerate(tokens):
                buffer += tk + " "
                if i % 8 == 0:
                    yield f"event: chunk\ndata: {json.dumps({'text': buffer})}\n\n"
                    buffer = ""
                    time.sleep(0.01)
            if buffer:
                yield f"event: chunk\ndata: {json.dumps({'text': buffer})}\n\n"
            yield "event: done\ndata: \n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async def reindex(self, req):
        pdf_path = req.pdf_path
        if not pdf_path:
            raise HTTPException(status_code=400, detail="pdf_path is required")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try:
            vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db", collection_name="testing")
        except TypeError:
            vectordb = Chroma.from_documents(chunks, embedding_function=embeddings, persist_directory="chroma_db", collection_name="testing")
        return {"status": "ok", "count": getattr(vectordb, "_collection", {}).count() if hasattr(vectordb, "_collection") else None}
