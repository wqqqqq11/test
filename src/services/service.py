from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
from ..utils.common import load_config, setup_logger
from ..models.models import CLIPEmbedder
from ..repositories.milvus_store import MilvusStore
from ..core.document_processor import DocumentProcessor
from pymilvus import utility

app = FastAPI(title="多语种向量检索服务")

config = load_config()
logger = setup_logger("Service", config)
embedder = CLIPEmbedder(config)
store = MilvusStore(config)
document_processor = DocumentProcessor(config)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total: int


class ProcessFilesRequest(BaseModel):
    file_paths: List[str]
    service_name: Optional[str] = ""
    user_name: Optional[str] = ""
    output_path: Optional[str] = None


class ProcessFilesResponse(BaseModel):
    success: bool
    message: str
    qa_count: int
    output_path: str


@app.on_event("startup")
async def startup():
    logger.info("启动服务")
    store.connect()
    # 确保集合存在（如果不存在则创建空集合）
    store.create_collection(drop_existing=False)
    store.load()
    logger.info("Milvus连接成功")


@app.on_event("shutdown")
async def shutdown():
    logger.info("关闭服务")
    store.disconnect()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "multilingual-vector-search"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        query_vector = embedder.encode([request.query])[0]
        
        results = store.search(
            query_vector=query_vector,
            top_k=min(request.top_k, 100)
        )
        
        return QueryResponse(
            results=results,
            query=request.query,
            total=len(results)
        )
    
    except Exception as e:
        logger.error(f"查询错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-files", response_model=ProcessFilesResponse)
async def process_files(request: ProcessFilesRequest):
    """处理文件并生成QA对数据集"""
    try:
        # 验证文件是否存在
        missing_files = [path for path in request.file_paths if not os.path.exists(path)]
        if missing_files:
            raise HTTPException(
                status_code=400, 
                detail=f"以下文件不存在: {missing_files}"
            )
        
        # 处理文件
        qa_data = document_processor.process_files(
            file_paths=request.file_paths,
            service_name=request.service_name,
            user_name=request.user_name
        )
        
        if not qa_data:
            return ProcessFilesResponse(
                success=False,
                message="未能生成任何QA对",
                qa_count=0,
                output_path=""
            )
        
        # 保存到CSV
        output_path = document_processor.save_to_csv(qa_data, request.output_path)
        
        return ProcessFilesResponse(
            success=True,
            message=f"成功处理 {len(request.file_paths)} 个文件",
            qa_count=len(qa_data),
            output_path=output_path
        )
        
    except Exception as e:
        logger.error(f"处理文件错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-uploaded-files", response_model=ProcessFilesResponse)
async def process_uploaded_files(
    files: List[UploadFile] = File(...),
    service_name: str = "",
    user_name: str = ""
):
    """处理上传的文件并生成QA对数据集"""
    try:
        # 准备文件数据
        uploaded_files = []
        for file in files:
            content = await file.read()
            uploaded_files.append({
                "name": file.filename,
                "content": content
            })
        
        # 处理文件
        qa_data = document_processor.process_uploaded_files(
            uploaded_files=uploaded_files,
            service_name=service_name,
            user_name=user_name
        )
        
        if not qa_data:
            return ProcessFilesResponse(
                success=False,
                message="未能生成任何QA对",
                qa_count=0,
                output_path=""
            )
        
        # 保存到CSV
        output_path = document_processor.save_to_csv(qa_data)
        
        return ProcessFilesResponse(
            success=True,
            message=f"成功处理 {len(files)} 个上传文件",
            qa_count=len(qa_data),
            output_path=output_path
        )
        
    except Exception as e:
        logger.error(f"处理上传文件错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def start_service():
    uvicorn.run(
        app,
        host=config['service']['host'],
        port=config['service']['port'],
        log_level="info"
    )


if __name__ == "__main__":
    start_service()
