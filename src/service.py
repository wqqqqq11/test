from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from .common import load_config, setup_logger
from .models import CLIPEmbedder
from .milvus_store import MilvusStore
from pymilvus import Collection

app = FastAPI(title="多语种向量检索服务")

config = load_config()
logger = setup_logger("Service", config)
embedder = CLIPEmbedder(config)
store = MilvusStore(config)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total: int


@app.on_event("startup")
async def startup():
    logger.info("启动服务")
    store.connect()
    # store.collection = store.Collection(config['milvus']['collection_name'])
    store.collection = Collection(config['milvus']['collection_name'])
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


def start_service():
    uvicorn.run(
        app,
        host=config['service']['host'],
        port=config['service']['port'],
        log_level="info"
    )


if __name__ == "__main__":
    start_service()
