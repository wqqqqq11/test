from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import pandas as pd
import io
from datetime import datetime
from ..utils.common import load_config, setup_logger
from ..models.models import CLIPEmbedder
from ..repositories.milvus_store import MilvusStore
from ..core.document_processor import DocumentProcessor
from ..core.pipeline import Pipeline
from ..core.qa_validator import QAValidator
from .test_services.qas_test_service import QASTestService, TestRequest, TestResponse
from .polish_service import PolishService
from pymilvus import utility

app = FastAPI(title="多语种向量检索服务")

config = load_config()
logger = setup_logger("Service", config)
embedder = CLIPEmbedder(config)
store = MilvusStore(config)
document_processor = DocumentProcessor(config)
pipeline = Pipeline()
test_service = QASTestService()
qa_validator = QAValidator(config)
polish_service = PolishService(config)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryResultItem(BaseModel):
    similarity_score: float
    generate_source: Optional[str] = ""
    question: str
    answer: str
    image_url: str

class CategoryItem(BaseModel):
    category_name: str
    items: List[QueryResultItem]

class QueryResponse(BaseModel):
    search_info: Dict[str, Any]
    categories: List[CategoryItem]


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


class VectorizeDatasetResponse(BaseModel):
    success: bool
    message: str
    total_records: int
    duration_seconds: float
    report_path: Optional[str] = None


class ValidationItem(BaseModel):
    row_id: Any
    question: str
    answer: str
    is_valid: bool
    reason: str


class ValidationResponse(BaseModel):
    success: bool
    message: str
    total_count: int
    valid_count: int
    invalid_count: int
    pass_rate: float
    output_path: Optional[str] = None
    results: List[ValidationItem]


class ProcessDocumentWithPolishResponse(BaseModel):
    success: bool
    message: str
    original_qa_count: int
    polished_qa_count: int
    vectorized_count: int
    validated_csv_path: str
    polished_csv_path: str
    vectorization_report: Optional[Dict[str, Any]] = None


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
        
        # 按类别分组结果
        category_dict = {}
        for result in results:
            category = result.get('category', '未分类')
            if not category or category.strip() == '':
                category = '未分类'
            
            # 将相似度分数转换为百分比格式
            similarity_score = round(result['score'] * 100, 2)
            
            query_item = QueryResultItem(
                similarity_score=similarity_score,
                generate_source=result.get('generate_source', ''),
                question=result.get('question', ''),
                answer=result.get('answer', ''),
                image_url=result.get('image_url', '')
            )
            
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(query_item)
        
        # 转换为数组格式
        categories = []
        for category_name, items in category_dict.items():
            categories.append(CategoryItem(
                category_name=category_name,
                items=items
            ))
        
        # 构建搜索信息
        search_info = {
            "query": request.query,
            "timestamp": datetime.now().isoformat(),
            "total_results": len(results)
        }
        
        return QueryResponse(
            search_info=search_info,
            categories=categories
        )
    
    except Exception as e:
        logger.error(f"查询错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-files", response_model=ProcessFilesResponse)
async def process_files(request: ProcessFilesRequest):
    """处理文件并生成QA对数据集（服务器）"""
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
    """处理上传的文件并生成QA对数据集（本地）"""
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


@app.post("/test/run-qa-test", response_model=TestResponse)
async def run_qa_test(request: TestRequest):
    """运行QA测试（服务器）"""
    try:
        return await test_service.run_test(request)
    except Exception as e:
        logger.error(f"QA测试错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test/run-qa-test-upload", response_model=TestResponse)
async def run_qa_test_with_upload(
    file: UploadFile = File(...),
    top_k: Optional[int] = None,
    recall_k_values: Optional[str] = None
):
    """使用上传文件运行QA测试"""
    try:
        # 解析recall_k_values参数
        k_values = None
        if recall_k_values:
            try:
                k_values = [int(k.strip()) for k in recall_k_values.split(',')]
            except ValueError:
                raise HTTPException(status_code=400, detail="recall_k_values格式错误，应为逗号分隔的数字")
        
        return await test_service.run_test_with_uploaded_file(file, top_k, k_values)
    except Exception as e:
        logger.error(f"上传文件QA测试错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectorize-dataset-upload", response_model=VectorizeDatasetResponse)
async def vectorize_dataset_upload(
    file: UploadFile = File(...),
    drop_existing: Optional[bool] = None
):
    """通过上传文件进行向量化存储"""
    temp_file_path = None
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="仅支持CSV文件")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = config.get('document_processing', {}).get('temp_dir', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"uploaded_dataset_{timestamp}.csv")
        
        content = await file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)
        
        if drop_existing is None:
            drop_existing = config.get('vectorization', {}).get('drop_existing', False)
        
        result = pipeline.vectorize_dataset(temp_file_path, drop_existing)
        
        report_path = None
        if result.get('report'):
            report_dir = config.get('vectorization', {}).get('report_output_dir', 'outputs/vectorization_reports')
            timestamp_str = result['report'].get('timestamp', timestamp)
            report_path = os.path.join(report_dir, f"performance_report_{timestamp_str}.json")
        
        return VectorizeDatasetResponse(
            success=True,
            message=f"成功向量化 {result['total_records']} 条记录",
            total_records=result['total_records'],
            duration_seconds=round(result['duration_seconds'], 2),
            report_path=report_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"向量化数据集错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"向量化失败: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


@app.post("/validate-qa-pairs", response_model=ValidationResponse)
async def validate_qa_pairs(file: UploadFile = File(...)):
    """校验CSV文件中的question和answer字段，删除不合格样本"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="仅支持CSV文件")
    
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), encoding='utf-8-sig')
    
    required_columns = ['question', 'answer']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"CSV文件缺少必要字段: {missing_columns}")
    
    id_column = 'ID' if 'ID' in df.columns else None
    results = []
    valid_indices = []
    valid_count = 0
    invalid_count = 0
    
    for idx, row in df.iterrows():
        question = str(row.get('question', '')).strip()
        answer = str(row.get('answer', '')).strip()
        row_id = row[id_column] if id_column else idx + 1
        
        is_valid, reason = qa_validator.validate_qa_pair(question, answer)
        
        if is_valid:
            valid_count += 1
            valid_indices.append(idx)
        else:
            invalid_count += 1
        
        results.append(ValidationItem(
            row_id=row_id,
            question=question,
            answer=answer,
            is_valid=is_valid,
            reason=reason
        ))
    
    total_count = len(results)
    pass_rate = (valid_count / total_count * 100) if total_count > 0 else 0
    
    output_path = None
    if valid_count > 0:
        valid_df = df.loc[valid_indices].copy()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('data', 'validated_data')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"validated_qa_pairs_{timestamp}.csv")
        valid_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return ValidationResponse(
        success=True,
        message=f"校验完成，共 {total_count} 条记录，删除 {invalid_count} 条不合格记录",
        total_count=total_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        pass_rate=round(pass_rate, 2),
        output_path=output_path,
        results=results
    )


@app.post("/process-document-with-polish", response_model=ProcessDocumentWithPolishResponse)
async def process_document_with_polish(
    file: UploadFile = File(...),
    service_name: str = "",
    user_name: str = "",
    is_stream: bool = False,
    drop_existing: bool = False
):
    """处理文档生成润色后的QA对并存入向量数据库"""
    temp_csv_path = None
    polished_csv_path = None
    
    uploaded_files = [{
        "name": file.filename,
        "content": await file.read()
    }]
    
    qa_data = document_processor.process_uploaded_files(
        uploaded_files=uploaded_files,
        service_name=service_name,
        user_name=user_name
    )
    
    if not qa_data:
        raise HTTPException(status_code=400, detail="未能生成任何QA对")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    validated_csv_path = os.path.join('data', f"validated_qa_{timestamp}.csv")
    os.makedirs('data', exist_ok=True)
    document_processor.save_to_csv(qa_data, validated_csv_path)
    
    polished_qa_data = await polish_service.polish_qa_pairs(qa_data)
    
    polished_csv_path = os.path.join('outputs', 'polished_data', f"polished_qa_{timestamp}.csv")
    os.makedirs(os.path.dirname(polished_csv_path), exist_ok=True)
    document_processor.save_to_csv(polished_qa_data, polished_csv_path)
    
    vectorization_result = pipeline.vectorize_dataset(polished_csv_path, drop_existing)
    
    if temp_csv_path and os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)
    
    return ProcessDocumentWithPolishResponse(
        success=True,
        message=f"处理完成，润色后生成 {len(polished_qa_data)} 条QA对",
        original_qa_count=len(qa_data),
        polished_qa_count=len(polished_qa_data),
        vectorized_count=vectorization_result['total_records'],
        validated_csv_path=validated_csv_path,
        polished_csv_path=polished_csv_path,
        vectorization_report=vectorization_result.get('report')
    )


def start_service():
    uvicorn.run(
        app,
        host=config['service']['host'],
        port=config['service']['port'],
        log_level="info"
    )


if __name__ == "__main__":
    start_service()
