import os
import re
import tempfile
import csv
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
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
from .qa_validator import QAValidator
from .data_tracer import DataTracer
from ..prompts.prompts import SALES_QA_GENERATION_PROMPT


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

    # -----------------------------
    # 结构合并（B档：文本 + 轻量统计）
    # -----------------------------
    _RE_NUM_TITLE = re.compile(r"^\s*\d{1,2}(\.\d{1,2}){0,4}\s+.+$")
    _RE_CN_SECTION = re.compile(r"^\s*(第[一二三四五六七八九十百千0-9]+[章节篇部分])\s*.+$")
    _RE_CN_ENUM = re.compile(r"^\s*([一二三四五六七八九十]+[、.])\s*.+$")
    _RE_BULLET = re.compile(r"^\s*([•\-\*·])\s+.+$")
    _RE_TOC_LEADER = re.compile(r".+\.{3,}\s*\d+\s*$")

    _TITLE_END_PUNCT = set("。！？.!?；;：:，,）)】]」』”\"")
    _LINE_TRIM_RE = re.compile(r"[ \t\u3000]+")  # normal + full-width spaces

    # -----------------------------
    # QA质量门控 + 产品关键词门控
    # -----------------------------
    _RE_ONLY_DIGITS = re.compile(r"^\s*[\d\W_]+\s*$")
    _RE_PAGE_NOISE = re.compile(r"^\s*(page|p)\s*\.?\s*\d+\s*$", re.IGNORECASE)
    _RE_MOSTLY_PUNCT = re.compile(r"^[\W_]+$")

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.doc_config = config["document_processing"]
        self.qwen_config = config["qwen"]
        self.logger = setup_logger("DocumentProcessor", config)

        self.client = OpenAI(
            api_key=self.qwen_config["api_key"],
            base_url=self.qwen_config["base_url"],
            timeout=self.qwen_config["timeout"],
        )

        os.makedirs(self.doc_config["temp_dir"], exist_ok=True)

        # 初始化QA校验器
        self.qa_validator = QAValidator(config)
        
        # 初始化数据追踪器
        self.data_tracer = DataTracer(config)

        # 产品聚焦关键词（可在 config 里覆盖/扩展）
        default_keywords = [
            "IRONFLIP",
            "AI",
            "离线",
            "语言模型",
            "翻译",
            "5G",
            "双模型",
            "隐私",
            "安全",
            "锁屏",
            "拍照",
            "屏幕",
            "外屏",
            "OLED",
            "刷新率",
            "亮度",
            "摄像头",
            "录音",
            "实时",
            "频段",
            "通信",
            "续航",
            "芯片",
            "防护",
            "材质",
            "工艺",
            "珐琅",
            "漆艺",
            "皮革",
            "配件",
            "价格",
            "售后",
            "质保",
            "场景",
            "对比",
            "优势",
        ]
        self.product_keywords = set(self.doc_config.get("product_keywords", default_keywords))

        # 质量门控参数（可在 config 中覆盖）
        self.min_chunk_chars = int(self.doc_config.get("min_chunk_chars", 120))
        self.max_digit_ratio = float(self.doc_config.get("max_digit_ratio", 0.45))
        self.max_punct_ratio = float(self.doc_config.get("max_punct_ratio", 0.55))
        self.keyword_min_hits = int(self.doc_config.get("keyword_min_hits", 1))

        # 销售场景：问题类型配额（可在 config 中调）
        self.sales_question_types = self.doc_config.get(
            "sales_question_types",
            [
                "核心卖点/差异化",
                "功能与使用场景",
                "规格参数与能力边界",
                "隐私安全与合规",
                "对比与选型建议",
                "常见疑虑与反驳",
                "操作引导/上手步骤",
                "售后/质保/交付",
            ],
        )

    # -----------------------------
    # 基础加载
    # -----------------------------
    def load_document(self, file_path: str, display_name: str = "") -> List[Document]:
        """加载单个文档"""
        ext = "." + file_path.rsplit(".", 1)[-1].lower()

        if ext not in self.LOADER_MAPPING:
            raise ValueError(f"不支持的文件扩展名: {ext}")

        loader_class, loader_args = self.LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()
        
        # 追踪原始文档数据
        trace_name = display_name if display_name else file_path
        self.data_tracer.trace_raw_documents(documents, trace_name)
        
        return documents

    # -----------------------------
    # 文本清洗/结构合并
    # -----------------------------
    def _normalize_lines(self, text: str) -> List[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines: List[str] = []
        for raw in text.split("\n"):
            s = raw.strip()
            if not s:
                continue
            s = self._LINE_TRIM_RE.sub(" ", s).strip()
            if s:
                lines.append(s)
        return lines

    def _detect_repeated_header_footer(
        self,
        pages_lines: List[List[str]],
        head_n: int = 3,
        tail_n: int = 3,
        min_pages: int = 3,
        ratio: float = 0.6,
        max_line_len: int = 40,
    ) -> Tuple[set, set]:
        if len(pages_lines) < min_pages:
            return set(), set()

        head_counter = Counter()
        tail_counter = Counter()

        for lines in pages_lines:
            head = [l for l in lines[:head_n] if 0 < len(l) <= max_line_len]
            tail = [l for l in lines[-tail_n:] if 0 < len(l) <= max_line_len]
            head_counter.update(head)
            tail_counter.update(tail)

        threshold = max(2, int(len(pages_lines) * ratio))
        repeated_head = {l for l, c in head_counter.items() if c >= threshold}
        repeated_tail = {l for l, c in tail_counter.items() if c >= threshold}
        return repeated_head, repeated_tail

    def _is_toc_page(self, lines: List[str]) -> bool:
        if not lines:
            return False

        joined = "\n".join(lines[:30])
        has_keyword = ("目录" in joined) or ("Contents" in joined) or ("CONTENTS" in joined)
        leader_hits = sum(1 for l in lines[:60] if self._RE_TOC_LEADER.match(l))

        if has_keyword and leader_hits >= 3:
            return True
        if leader_hits >= 8:
            return True
        return False

    def _looks_like_title_line(
        self,
        line: str,
        prev_blank: bool,
        next_blank: bool,
        typical_len: int = 22,
        max_len: int = 60,
    ) -> bool:
        if not line:
            return False

        L = len(line)
        if L > max_len:
            return False

        if self._RE_BULLET.match(line):
            return False

        strong = bool(
            self._RE_NUM_TITLE.match(line)
            or self._RE_CN_SECTION.match(line)
            or self._RE_CN_ENUM.match(line)
        )

        punct_cnt = sum(1 for ch in line if ch in self._TITLE_END_PUNCT)
        few_punct = punct_cnt <= 1
        not_sentence_end = (line[-1] not in self._TITLE_END_PUNCT) or (line[-1] in "：:")
        shortish = L <= typical_len or (L <= max_len and few_punct)
        boundary_bonus = prev_blank or next_blank

        if strong:
            return True

        if shortish and few_punct and not_sentence_end and boundary_bonus:
            return True

        if L <= 12 and few_punct and not_sentence_end:
            return True

        return False

    def merge_pages_by_structure(
        self,
        documents: List[Document],
        drop_toc_pages: bool = True,
        head_n: int = 3,
        tail_n: int = 3,
    ) -> List[Document]:
        """
        将按页的 Document 合并为“按章节/小节”的 Document，保证跨页上下文连续。
        """
        if not documents:
            return []

        pages_lines: List[List[str]] = []
        pages_raw_lines: List[List[str]] = []

        for d in documents:
            raw = d.page_content if hasattr(d, "page_content") else str(d)
            raw = raw.replace("\r\n", "\n").replace("\r", "\n")
            raw_lines = [self._LINE_TRIM_RE.sub(" ", x.rstrip()) for x in raw.split("\n")]
            pages_raw_lines.append(raw_lines)
            pages_lines.append(self._normalize_lines(raw))

        repeated_head, repeated_tail = self._detect_repeated_header_footer(
            pages_lines, head_n=head_n, tail_n=tail_n
        )

        merged: List[Document] = []
        current_lines: List[str] = []
        current_meta: Dict[str, Any] = {
            "section_title": None,
            "start_page": 1,
            "end_page": 1,
            "source_pages": [],
        }

        def flush():
            nonlocal current_lines, current_meta, merged
            text = "\n".join(current_lines).strip()
            if not text:
                current_lines = []
                return
            merged.append(Document(page_content=text, metadata=dict(current_meta)))
            current_lines = []

        for page_idx, raw_lines in enumerate(pages_raw_lines):
            page_num = page_idx + 1
            normalized_for_toc = self._normalize_lines("\n".join(raw_lines))
            if drop_toc_pages and self._is_toc_page(normalized_for_toc):
                continue

            cleaned: List[Tuple[str, bool]] = []
            for rl in raw_lines:
                s = self._LINE_TRIM_RE.sub(" ", rl).strip()
                is_blank = (s == "")
                if not is_blank:
                    if s in repeated_head or s in repeated_tail:
                        continue
                cleaned.append((s, is_blank))

            compact: List[Tuple[str, bool]] = []
            prev_blank = False
            for s, is_blank in cleaned:
                if is_blank:
                    if not prev_blank:
                        compact.append(("", True))
                    prev_blank = True
                else:
                    compact.append((s, False))
                    prev_blank = False

            for i, (s, is_blank) in enumerate(compact):
                if is_blank:
                    continue

                prev_is_blank = True
                next_is_blank = True
                if i - 1 >= 0:
                    prev_is_blank = compact[i - 1][1]
                if i + 1 < len(compact):
                    next_is_blank = compact[i + 1][1]

                if self._looks_like_title_line(s, prev_blank=prev_is_blank, next_blank=next_is_blank):
                    if current_lines:
                        flush()
                        current_meta = {
                            "section_title": s,
                            "start_page": page_num,
                            "end_page": page_num,
                            "source_pages": [page_num],
                        }
                    else:
                        current_meta["section_title"] = s
                        current_meta["start_page"] = page_num
                        current_meta["end_page"] = page_num
                        if not current_meta.get("source_pages"):
                            current_meta["source_pages"] = [page_num]

                    current_lines.append(s)
                    continue

                if not current_lines:
                    current_meta = {
                        "section_title": current_meta.get("section_title") or "前言",
                        "start_page": page_num,
                        "end_page": page_num,
                        "source_pages": [page_num],
                    }
                else:
                    current_meta["end_page"] = page_num
                    if page_num not in current_meta["source_pages"]:
                        current_meta["source_pages"].append(page_num)

                current_lines.append(s)

        if current_lines:
            flush()

        return merged

    # -----------------------------
    # chunk 质量门控 + 产品关键词门控
    # -----------------------------
    def _chunk_quality_ok(self, text: str) -> bool:
        if not text:
            return False

        t = text.strip()
        if len(t) < self.min_chunk_chars:
            return False

        # 常见噪声：纯数字/符号，或页码形式
        if self._RE_ONLY_DIGITS.match(t):
            return False
        if self._RE_PAGE_NOISE.match(t):
            return False
        if self._RE_MOSTLY_PUNCT.match(t):
            return False

        total = len(t)
        if total <= 0:
            return False

        digit_cnt = sum(ch.isdigit() for ch in t)
        punct_cnt = sum((not ch.isalnum()) and (not ("\u4e00" <= ch <= "\u9fff")) and (not ch.isspace()) for ch in t)

        digit_ratio = digit_cnt / total
        punct_ratio = punct_cnt / total

        if digit_ratio > self.max_digit_ratio:
            return False
        if punct_ratio > self.max_punct_ratio:
            return False

        # 至少包含一定的可读文字
        cn_cnt = sum("\u4e00" <= ch <= "\u9fff" for ch in t)
        en_cnt = sum(("a" <= ch.lower() <= "z") for ch in t)
        if (cn_cnt + en_cnt) < 30:
            return False

        return True

    def _product_keyword_hits(self, text: str) -> int:
        if not text:
            return 0
        hits = 0
        low = text.lower()
        for kw in self.product_keywords:
            if not kw:
                continue
            if kw.lower() in low:
                hits += 1
        return hits

    def _should_generate_sales_qa(self, chunk_text: str) -> bool:
        if not self._chunk_quality_ok(chunk_text):
            return False

        hits = self._product_keyword_hits(chunk_text)
        if hits < self.keyword_min_hits:
            return False

        return True

    # -----------------------------
    # 分割：PDF 先结构合并，保证跨页上下文；再 chunking
    # -----------------------------
    def split_documents(self, documents: List[Document], ext: str = "", file_path: str = "") -> List[Document]:
        """将文档分割成文本块"""
        # 追踪清洗前的文档
        if file_path:
            self.data_tracer.trace_cleaned_documents(documents, file_path)
        
        merged_documents = self.merge_pages_by_structure(
            documents,
            drop_toc_pages=True,
            head_n=int(self.doc_config.get("header_lines", 3)),
            tail_n=int(self.doc_config.get("footer_lines", 3)),
        )
        
        # 追踪结构合并后的文档
        if file_path:
            self.data_tracer.trace_merged_documents(merged_documents, file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.doc_config["chunk_size"]),
            chunk_overlap=int(self.doc_config["chunk_overlap"]),
        )
        text_chunks = text_splitter.split_documents(merged_documents)
        
        # 追踪文本分块数据
        if file_path:
            self.data_tracer.trace_text_chunks(text_chunks, file_path)
        
        return text_chunks

    # -----------------------------
    # 销售场景 QA 生成（强约束）
    # -----------------------------
    def _build_sales_prompt(self, chunk_text: str, max_pairs: int, section_title: str = "") -> str:
        # 通过“明确任务 + 明确约束 + 明确输出格式”来稳定产出
        question_type_hint = "\n".join([f"- {x}" for x in self.sales_question_types])

        header = []
        if section_title:
            header.append(f"章节/小节标题：{section_title}")
        header.append(f"最多生成：{max_pairs} 组问答")

        header_text = "\n".join(header)

        return SALES_QA_GENERATION_PROMPT.format(
            max_pairs=max_pairs,
            header_text=header_text,
            chunk_text=chunk_text
        )


    def generate_qa_pairs(self, text_chunk: Document) -> List[Dict[str, str]]:
        """为单个文本块生成销售场景 QA 对（带质量/关键词门控）"""
        chunk_text = text_chunk.page_content if hasattr(text_chunk, "page_content") else str(text_chunk)

        # 1) 质量门控 + 2) 产品关键词门控
        if not self._should_generate_sales_qa(chunk_text):
            return []

        max_pairs = int(self.doc_config["max_qa_pairs_per_chunk"])
        section_title = ""
        if hasattr(text_chunk, "metadata") and isinstance(text_chunk.metadata, dict):
            section_title = str(text_chunk.metadata.get("section_title") or "")

        prompt = self._build_sales_prompt(chunk_text=chunk_text, max_pairs=max_pairs, section_title=section_title)

        try:
            response = self.client.chat.completions.create(
                model=self.qwen_config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            response_text = (response.choices[0].message.content or "").strip()
            if not response_text:
                return []

            qa_pairs: List[Dict[str, str]] = []
            blocks = [b.strip() for b in response_text.split("\n\n") if b.strip()]
            for block in blocks:
                # 允许模型多输出空行，但要求必须同时含 Q: 和 A:
                if "Q:" not in block or "A:" not in block:
                    continue

                # 优先按 A: 切分
                parts = block.split("A:", 1)
                if len(parts) != 2:
                    continue

                q_part = parts[0]
                a_part = parts[1]

                question = q_part.replace("Q:", "").strip()
                answer = a_part.strip()

                # 再次兜底过滤：避免“文本是什么/是否有意义”
                if not question or not answer:
                    continue
                bad_q = ("文本" in question and ("是什么" in question or "有意义" in question)) or ("这段" in question and "讲" in question)
                if bad_q:
                    continue

                qa_pairs.append({"question": question, "answer": answer})

                if len(qa_pairs) >= max_pairs:
                    break

            return qa_pairs

        except Exception as e:
            self.logger.error(f"生成QA对时出错: {str(e)}")
            return []

    # -----------------------------
    # 主流程
    # -----------------------------
    def process_file(
        self,
        file_path: str,
        service_name: str = "",
        user_name: str = "",
        start_id: int = 1,
        display_name: str = "",
        manage_session: bool = True,
    ) -> List[Dict[str, Any]]:
        """处理单个文件并生成QA对数据"""
        try:
            ext = "." + file_path.rsplit(".", 1)[-1].lower()
            trace_name = display_name if display_name else file_path

            # 开始追踪会话（如果需要）
            if manage_session:
                self.data_tracer.start_session(f"process_file_{os.path.basename(trace_name)}")

            documents = self.load_document(file_path, trace_name)
            if not documents:
                self.logger.error(f"无法加载文档: {file_path}")
                return []

            # PDF：先结构合并（跨页）再切 chunk；其他类型：直接切 chunk
            text_chunks = self.split_documents(documents, ext=ext, file_path=trace_name)
            self.logger.info(f"文档 {file_path} 已分割成 {len(text_chunks)} 个文本块")

            all_qa_data: List[Dict[str, Any]] = []
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_date = datetime.now().strftime("%Y-%m-%d")

            qa_id = start_id
            svc = service_name if service_name else "AI销售"
            usr = user_name if user_name else "客户A"

            for chunk in text_chunks:
                qa_pairs = self.generate_qa_pairs(chunk)
                if not qa_pairs:
                    continue

                # 追踪生成的QA对数据
                self.data_tracer.trace_generated_qa(qa_pairs, chunk.page_content, trace_name)

                # 校验生成的QA对
                validated_qa_pairs = self.qa_validator.validate_qa_batch(qa_pairs)
                if not validated_qa_pairs:
                    self.logger.info(f"该文本块生成的所有QA对都未通过校验，跳过")
                    continue

                # 追踪校验后的QA对数据
                self.data_tracer.trace_validated_qa(qa_pairs, validated_qa_pairs, trace_name)

                for qa_pair in validated_qa_pairs:
                    qa_data = {
                        "ID": qa_id,
                        "service_name": svc,
                        "user_name": usr,
                        "question_time": current_time,
                        "data": current_date,
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRVuBc3P2Xjkcux2LzpUDN6dS2vYIngfCGaIwiri8KalXJrH4gw25HwzxqI&s",
                    }
                    all_qa_data.append(qa_data)
                    qa_id += 1
                # if qa_id == 1:
                #     break

            self.logger.info(f"为文档 {file_path} 生成了 {len(all_qa_data)} 个QA对")
            
            # 结束追踪会话（如果需要）
            if manage_session:
                self.data_tracer.end_session()
            
            return all_qa_data

        except Exception as e:
            self.logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            # 确保在异常情况下也结束会话
            if manage_session:
                self.data_tracer.end_session()
            return []

    def process_files(self, file_paths: List[str], service_name: str = "", user_name: str = "") -> List[Dict[str, Any]]:
        """处理多个文件并生成QA对数据"""
        all_qa_data: List[Dict[str, Any]] = []
        current_id = 1

        for file_path in file_paths:
            if not os.path.exists(file_path):
                self.logger.warning(f"文件不存在: {file_path}")
                continue

            qa_data = self.process_file(file_path, service_name, user_name, current_id)
            all_qa_data.extend(qa_data)
            current_id += len(qa_data)

        return all_qa_data

    def process_uploaded_files(
        self,
        uploaded_files: List[Dict[str, Any]],
        service_name: str = "",
        user_name: str = "",
    ) -> List[Dict[str, Any]]:
        """处理上传的文件（字节流）并生成QA对数据"""
        all_qa_data: List[Dict[str, Any]] = []
        current_id = 1

        # 开始批量处理追踪会话
        self.data_tracer.start_session("process_uploaded_files_batch")

        try:
            for file_info in uploaded_files:
                file_name = file_info["name"]
                file_content = file_info["content"]

                ext = os.path.splitext(file_name)[1].lower()
                if ext not in self.doc_config["supported_extensions"]:
                    self.logger.warning(f"不支持的文件类型: {file_name}")
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=self.doc_config["temp_dir"]) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name

                try:
                    qa_data = self.process_file(tmp_file_path, service_name, user_name, current_id, file_name, manage_session=False)
                    all_qa_data.extend(qa_data)
                    current_id += len(qa_data)
                finally:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

        finally:
            # 结束批量处理追踪会话
            self.data_tracer.end_session()

        return all_qa_data

    def save_to_csv(self, qa_data: List[Dict[str, Any]], output_path: str = None) -> str:
        """将QA对数据保存为CSV文件"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed_qa_pairs_{timestamp}.csv"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fieldnames = ["ID", "service_name", "user_name", "question_time", "data", "question", "answer", "image_url"]

        with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(qa_data)

        self.logger.info(f"已保存 {len(qa_data)} 条QA对数据到 {output_path}")
        return output_path

    def get_validation_preview(self, file_path: str, max_chunks: int = 3) -> Dict[str, Any]:
        """
        预览文档的QA生成和校验情况（用于快速评估）
        
        Args:
            file_path: 文档路径
            max_chunks: 最多处理的文本块数量
            
        Returns:
            Dict[str, Any]: 包含预览统计信息
        """
        try:
            ext = "." + file_path.rsplit(".", 1)[-1].lower()
            documents = self.load_document(file_path)
            if not documents:
                return {"error": f"无法加载文档: {file_path}"}

            text_chunks = self.split_documents(documents, ext=ext, file_path=file_path)
            
            preview_stats = {
                "file_path": file_path,
                "total_chunks": len(text_chunks),
                "processed_chunks": min(max_chunks, len(text_chunks)),
                "generated_qa_pairs": 0,
                "validated_qa_pairs": 0,
                "validation_details": [],
                "sample_qa_pairs": []
            }
            
            for i, chunk in enumerate(text_chunks[:max_chunks]):
                qa_pairs = self.generate_qa_pairs(chunk)
                if not qa_pairs:
                    continue
                
                preview_stats["generated_qa_pairs"] += len(qa_pairs)
                
                # 获取校验统计
                validation_stats = self.qa_validator.get_validation_stats(qa_pairs)
                preview_stats["validation_details"].append({
                    "chunk_index": i,
                    "stats": validation_stats
                })
                
                # 实际校验一小部分作为样本
                validated_pairs = self.qa_validator.validate_qa_batch(qa_pairs[:2])  # 只校验前2个
                preview_stats["validated_qa_pairs"] += len(validated_pairs)
                
                # 保存样本
                for vp in validated_pairs[:1]:  # 只保存1个样本
                    preview_stats["sample_qa_pairs"].append({
                        "chunk_index": i,
                        "question": vp["question"],
                        "answer": vp["answer"][:100] + "..." if len(vp["answer"]) > 100 else vp["answer"]
                    })
            
            return preview_stats
            
        except Exception as e:
            self.logger.error(f"预览文档 {file_path} 时出错: {str(e)}")
            return {"error": str(e)}
