import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.documents import Document

from ..utils.common import setup_logger


class DataTracer:
    """数据追踪器，用于保存处理流程中各阶段的数据样本"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trace_config = config.get("trace", {})
        self.enabled = self.trace_config.get("enable_tracing", False)
        self.trace_dir = self.trace_config.get("trace_dir", "data/trace_data")
        self.sample_size = self.trace_config.get("sample_size", 5)
        self.stages = self.trace_config.get("stages", {})
        self.logger = setup_logger(self.__class__.__name__, config)
        
        # 会话数据存储
        self.session_id = None
        self.session_data = {}
        
        if self.enabled:
            os.makedirs(self.trace_dir, exist_ok=True)
    
    def start_session(self, session_name: str = "") -> None:
        """开始新的追踪会话"""
        if not self.enabled:
            return
            
        self.session_id = self._get_timestamp()
        self.session_data = {
            "session_id": self.session_id,
            "session_name": session_name,
            "start_time": datetime.now().isoformat(),
            "stages": {}
        }
        self.logger.info(f"开始数据追踪会话: {self.session_id}")
    
    def end_session(self) -> None:
        """结束当前追踪会话并保存所有数据"""
        if not self.enabled or not self.session_id:
            return
            
        self.session_data["end_time"] = datetime.now().isoformat()
        
        # 保存会话数据
        filename = f"trace_session_{self.session_id}.json"
        filepath = os.path.join(self.trace_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"追踪会话数据已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"保存追踪会话数据失败 {filepath}: {str(e)}")
        
        # 重置会话
        self.session_id = None
        self.session_data = {}
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _sample_data(self, data: List[Any], stage: str) -> List[Any]:
        """对数据进行采样"""
        if not data or len(data) <= self.sample_size:
            return data
        
        step = len(data) // self.sample_size
        sampled = []
        for i in range(0, len(data), step):
            if len(sampled) < self.sample_size:
                sampled.append(data[i])
        return sampled
    
    def _add_to_session(self, stage: str, data: Dict[str, Any]) -> None:
        """将数据添加到当前会话"""
        if not self.enabled or not self.session_id:
            return
            
        if stage not in self.session_data["stages"]:
            self.session_data["stages"][stage] = []
        
        self.session_data["stages"][stage].append(data)
    
    def trace_raw_documents(self, documents: List[Document], file_path: str) -> None:
        """追踪原始文档数据"""
        if not self.enabled or not self.stages.get("raw_documents", False):
            return
        
        sampled_docs = self._sample_data(documents, "raw_documents")
        trace_data = {
            "timestamp": self._get_timestamp(),
            "file_path": file_path,
            "total_count": len(documents),
            "sample_count": len(sampled_docs),
            "documents": [
                {
                    "page_content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in sampled_docs
            ]
        }
        
        self._add_to_session("raw_documents", trace_data)
    
    def trace_cleaned_documents(self, documents: List[Document], file_path: str) -> None:
        """追踪清洗后的文档数据"""
        if not self.enabled or not self.stages.get("cleaned_documents", False):
            return
        
        sampled_docs = self._sample_data(documents, "cleaned_documents")
        trace_data = {
            "timestamp": self._get_timestamp(),
            "file_path": file_path,
            "total_count": len(documents),
            "sample_count": len(sampled_docs),
            "documents": [
                {
                    "page_content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in sampled_docs
            ]
        }
        
        self._add_to_session("cleaned_documents", trace_data)
    
    def trace_merged_documents(self, documents: List[Document], file_path: str) -> None:
        """追踪结构合并后的文档数据"""
        if not self.enabled or not self.stages.get("merged_documents", False):
            return
        
        sampled_docs = self._sample_data(documents, "merged_documents")
        trace_data = {
            "timestamp": self._get_timestamp(),
            "file_path": file_path,
            "total_count": len(documents),
            "sample_count": len(sampled_docs),
            "documents": [
                {
                    "page_content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in sampled_docs
            ]
        }
        
        self._add_to_session("merged_documents", trace_data)
    
    def trace_text_chunks(self, chunks: List[Document], file_path: str) -> None:
        """追踪文本分块数据"""
        if not self.enabled or not self.stages.get("text_chunks", False):
            return
        
        sampled_chunks = self._sample_data(chunks, "text_chunks")
        trace_data = {
            "timestamp": self._get_timestamp(),
            "file_path": file_path,
            "total_count": len(chunks),
            "sample_count": len(sampled_chunks),
            "chunks": [
                {
                    "content": chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content,
                    "metadata": chunk.metadata,
                    "length": len(chunk.page_content)
                }
                for chunk in sampled_chunks
            ]
        }
        
        self._add_to_session("text_chunks", trace_data)
    
    def trace_generated_qa(self, qa_pairs: List[Dict[str, str]], chunk_content: str, file_path: str) -> None:
        """追踪生成的问答对数据"""
        if not self.enabled or not self.stages.get("generated_qa", False):
            return
        
        trace_data = {
            "timestamp": self._get_timestamp(),
            "file_path": file_path,
            "chunk_content": chunk_content[:300] + "..." if len(chunk_content) > 300 else chunk_content,
            "qa_count": len(qa_pairs),
            "qa_pairs": qa_pairs
        }
        
        self._add_to_session("generated_qa", trace_data)
    
    def trace_validated_qa(self, original_qa: List[Dict[str, str]], validated_qa: List[Dict[str, str]], file_path: str) -> None:
        """追踪校验后的问答对数据"""
        if not self.enabled or not self.stages.get("validated_qa", False):
            return
        
        trace_data = {
            "timestamp": self._get_timestamp(),
            "file_path": file_path,
            "original_count": len(original_qa),
            "validated_count": len(validated_qa),
            "validation_rate": len(validated_qa) / len(original_qa) if original_qa else 0,
            "original_qa": original_qa,
            "validated_qa": validated_qa,
            "rejected_qa": [qa for qa in original_qa if qa not in validated_qa]
        }
        
        self._add_to_session("validated_qa", trace_data)