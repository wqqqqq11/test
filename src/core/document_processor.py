import os
import tempfile
import csv
from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from ..utils.common import setup_logger


class DocumentProcessor:
    """文档处理器，负责加载文档、分块和生成QA对"""
    
    LOADER_MAPPING = {
        ".csv": (CSVLoader, {}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".enex": (EverNoteLoader, {}),
        ".eml": (UnstructuredEmailLoader, {}),
        ".epub": (UnstructuredEPubLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".odt": (UnstructuredODTLoader, {}),
        ".pdf": (PyMuPDFLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.doc_config = config['document_processing']
        self.qwen_config = config['qwen']
        self.logger = setup_logger("DocumentProcessor", config)
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.qwen_config['api_key'],
            base_url=self.qwen_config['base_url'],
            timeout=self.qwen_config['timeout']
        )
        
        # 创建临时目录
        os.makedirs(self.doc_config['temp_dir'], exist_ok=True)
        
    def load_document(self, file_path: str) -> List[Document]:
        """加载单个文档"""
        ext = "." + file_path.rsplit(".", 1)[-1].lower()
        
        if ext not in self.LOADER_MAPPING:
            raise ValueError(f"不支持的文件扩展名: {ext}")
            
        loader_class, loader_args = self.LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割成文本块"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.doc_config['chunk_size'],
            chunk_overlap=self.doc_config['chunk_overlap']
        )
        return text_splitter.split_documents(documents)
    
    def generate_qa_pairs(self, text_chunk: Document) -> List[Dict[str, str]]:
        """为单个文本块生成QA对"""
        chunk_text = text_chunk.page_content if hasattr(text_chunk, 'page_content') else str(text_chunk)
        
        max_pairs = self.doc_config['max_qa_pairs_per_chunk']
        prompt = f"""你是一名专业的问答数据构建助手，擅长从非结构化文本中抽取关键信息并生成高质量问答对。

任务说明：
请基于下方提供的文本内容，生成不超过 {max_pairs} 个问答对，用于知识问答或模型训练。

生成要求：
1. 问题需使用不同表述方式，覆盖文本中的关键信息（如产品功能、特点、用途、规格、优势、限制等）
2. 每个答案必须严格基于原文内容，准确、完整，不得引入原文中未提及的信息
3. 问题应清晰、自然，答案应简洁但信息充分
4. 如果文本信息较少或高度重复，可只生成 1–2 个高质量问答对
5. 不要对文本内容进行总结或改写，只生成问答对

输出格式要求：
- 每个问答对使用以下格式：
  Q: 问题
  A: 答案
- 问答对之间使用一个空行分隔
- 只输出 Q 和 A 的内容，不要添加编号、标题或任何额外说明

文本内容：
{chunk_text}

请生成问答对（最多 {max_pairs} 个）：
"""

        try:
            response = self.client.chat.completions.create(
                model=self.qwen_config['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            
            qa_pairs = []
            response_text = response.choices[0].message.content.strip()
            
            # 解析多个QA对，按空行分割
            qa_blocks = response_text.split('\n\n')
            for qa_block in qa_blocks:
                qa_block = qa_block.strip()
                if not qa_block:
                    continue
                    
                # 按A:分割，获取问题和答案
                parts = qa_block.split("A:", 1)
                if len(parts) == 2:
                    question = parts[0].replace("Q:", "").strip()
                    answer = parts[1].strip()
                    if question and answer:
                        qa_pairs.append({"question": question, "answer": answer})
                else:
                    # 尝试按Q:分割
                    parts = qa_block.split("Q:", 1)
                    if len(parts) == 2:
                        remaining = parts[1]
                        if "A:" in remaining:
                            qa_parts = remaining.split("A:", 1)
                            if len(qa_parts) == 2:
                                question = qa_parts[0].strip()
                                answer = qa_parts[1].strip()
                                if question and answer:
                                    qa_pairs.append({"question": question, "answer": answer})
            
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"生成QA对时出错: {str(e)}")
            return []
    
    def process_file(self, file_path: str, service_name: str = "", user_name: str = "", start_id: int = 1) -> List[Dict[str, Any]]:
        """处理单个文件并生成QA对数据"""
        try:
            # 加载文档
            documents = self.load_document(file_path)
            if not documents:
                self.logger.error(f"无法加载文档: {file_path}")
                return []
            
            # 分割文档
            text_chunks = self.split_documents(documents)
            self.logger.info(f"文档 {file_path} 已分割成 {len(text_chunks)} 个文本块")
            
            # 生成QA对
            all_qa_data = []
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_date = datetime.now().strftime("%Y-%m-%d")
            qa_id = start_id
            
            for i, chunk in enumerate(text_chunks):
                qa_pairs = self.generate_qa_pairs(chunk)
                
                for qa_pair in qa_pairs:
                    qa_data = {
                        "ID": qa_id,
                        "service_name": "AI销售",
                        "user_name": "客户A",
                        "question_time": current_time,
                        "data": current_date,
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRVuBc3P2Xjkcux2LzpUDN6dS2vYIngfCGaIwiri8KalXJrH4gw25HwzxqI&s"
                    }
                    all_qa_data.append(qa_data)
                    qa_id += 1
                if qa_id == 3:
                    break
            
            self.logger.info(f"为文档 {file_path} 生成了 {len(all_qa_data)} 个QA对")
            return all_qa_data
            
        except Exception as e:
            self.logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            return []
    
    def process_files(self, file_paths: List[str], service_name: str = "", user_name: str = "") -> List[Dict[str, Any]]:
        """处理多个文件并生成QA对数据"""
        all_qa_data = []
        current_id = 1
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                self.logger.warning(f"文件不存在: {file_path}")
                continue
                
            qa_data = self.process_file(file_path, service_name, user_name, current_id)
            all_qa_data.extend(qa_data)
            current_id += len(qa_data)
        
        return all_qa_data
    
    def process_uploaded_files(self, uploaded_files: List[Dict[str, Any]], service_name: str = "", user_name: str = "") -> List[Dict[str, Any]]:
        """处理上传的文件（字节流）并生成QA对数据"""
        all_qa_data = []
        current_id = 1
        
        for file_info in uploaded_files:
            file_name = file_info['name']
            file_content = file_info['content']
            
            # 获取文件扩展名
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in self.doc_config['supported_extensions']:
                self.logger.warning(f"不支持的文件类型: {file_name}")
                continue
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=self.doc_config['temp_dir']) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                qa_data = self.process_file(tmp_file_path, service_name, user_name, current_id)
                all_qa_data.extend(qa_data)
                current_id += len(qa_data)
            finally:
                # 清理临时文件
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        return all_qa_data
    
    def save_to_csv(self, qa_data: List[Dict[str, Any]], output_path: str = None) -> str:
        """将QA对数据保存为CSV文件"""
        if not output_path:
            # 基于当前时间戳生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed_qa_pairs_{timestamp}.csv"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fieldnames = ["ID", "service_name", "user_name", "question_time", "data", "question", "answer", "image_url"]
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(qa_data)
        
        self.logger.info(f"已保存 {len(qa_data)} 条QA对数据到 {output_path}")
        return output_path