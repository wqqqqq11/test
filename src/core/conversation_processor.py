"""
对话数据处理核心模块
处理客服对话数据，提取有效问答对并生成训练数据
"""

import json
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import asyncio

from ..services.llm_service import LLMService
from ..services.intent_service import IntentService
from ..services.polish_service import PolishService
from ..repositories.milvus_store import MilvusStore


class ConversationProcessor:
    """对话数据处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.conv_config = config['conversation_processing']
        self.logger = logging.getLogger(__name__)
        
        # 初始化服务
        self.llm_service = LLMService(config)
        self.intent_service = IntentService(config)
        self.polish_service = PolishService(config)
        self.vector_store = MilvusStore(config)
        
        # 滑动窗口大小
        self.window_size = self.conv_config['sliding_window_size']
        
        # 会话级别的参数对队列
        self.session_queues: Dict[str, deque] = {}
    
    async def process_conversations(self) -> None:
        """
        处理所有对话数据的主入口
        """
        self.logger.info("开始处理对话数据")
        
        # 读取输入数据
        input_file = self.conv_config['input_file']
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 准备输出文件
        output_file = self.conv_config['output_csv']
        processed_records = []
        
        # 按会话处理
        for record in data.get('RECORDS', []):
            session_id = str(record.get('ID', ''))
            service_name = record.get('客服名称', '')
            user_name = record.get('用户名称', '')
            messages = record.get('消息内容', [])
            
            self.logger.info(f"处理会话 {session_id}")
            
            # 处理单个会话
            session_results = await self._process_single_session(
                session_id, service_name, user_name, messages
            )
            
            processed_records.extend(session_results)
        
        # 写入CSV文件
        self._write_to_csv(processed_records, output_file)
        
        self.logger.info(f"处理完成，共生成 {len(processed_records)} 条记录")
    
    async def _process_single_session(
        self, 
        session_id: str, 
        service_name: str, 
        user_name: str, 
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        处理单个会话
        
        Args:
            session_id: 会话ID
            service_name: 客服名称
            user_name: 用户名称
            messages: 消息列表
            
        Returns:
            处理后的记录列表
        """
        results = []
        
        # 初始化会话队列
        if session_id not in self.session_queues:
            self.session_queues[session_id] = deque(maxlen=self.window_size)
        
        # 按时间排序消息
        sorted_messages = sorted(messages, key=lambda x: x.get('time', ''))
        
        # 遍历消息，查找客服回复
        for i, message in enumerate(sorted_messages):
            if message.get('sender') == '客服':
                content = message.get('content', '').strip()
                time_str = message.get('time', '')
                
                # 步骤3: 语义有效性校验
                is_meaningful = await self.llm_service.validate_semantic_meaning(content)
                if not is_meaningful:
                    self.logger.debug(f"跳过无意义客服回复: {content[:50]}...")
                    continue
                
                # 步骤4: 回溯并筛选相关客户问题
                related_questions = await self._extract_related_questions(
                    sorted_messages[:i], content
                )
                
                if not related_questions:
                    self.logger.debug(f"未找到相关客户问题，跳过回复: {content[:50]}...")
                    continue
                
                # 步骤5: 问题融合
                fused_question = await self.llm_service.fuse_questions(related_questions)
                if not fused_question:
                    self.logger.warning(f"问题融合失败，跳过")
                    continue
                
                # 添加到会话队列
                param_pair = (fused_question, content)
                self.session_queues[session_id].append(param_pair)
                
                # 步骤6: 意图改写（滑动窗口策略）
                intent_question = await self._process_intent_rewrite(session_id)
                if not intent_question:
                    self.logger.warning(f"意图改写失败，使用原问题")
                    intent_question = fused_question
                
                # 步骤7: 向量检索
                candidate_answer = await self._vector_search(intent_question)
                if not candidate_answer:
                    self.logger.warning(f"向量检索未找到候选答案，使用原答案")
                    candidate_answer = content
                
                # 步骤8: 回答润色
                final_answer = await self._polish_answer(intent_question, candidate_answer)
                if not final_answer:
                    self.logger.warning(f"回答润色失败，使用候选答案")
                    final_answer = candidate_answer
                
                # 步骤9: 构造结果记录
                record = self._build_result_record(
                    session_id, service_name, user_name, time_str,
                    intent_question, final_answer
                )
                results.append(record)
                
                self.logger.debug(f"成功处理一条记录: Q={intent_question[:50]}...")
        
        return results
    
    async def _extract_related_questions(
        self, 
        previous_messages: List[Dict[str, Any]], 
        answer: str
    ) -> List[str]:
        """
        提取与答案相关的客户问题
        
        Args:
            previous_messages: 之前的消息列表
            answer: 客服回答
            
        Returns:
            相关问题列表
        """
        related_questions = []
        
        # 获取所有客户消息
        customer_messages = [
            msg for msg in previous_messages 
            if msg.get('sender') == '客户'
        ]
        
        # 逐一判断相关性
        for msg in customer_messages:
            question = msg.get('content', '').strip()
            if not question:
                continue
                
            is_related = await self.llm_service.check_relevance(question, answer)
            if is_related:
                related_questions.append(question)
        
        return related_questions
    
    async def _process_intent_rewrite(self, session_id: str) -> Optional[str]:
        """
        处理意图改写（滑动窗口策略）
        
        Args:
            session_id: 会话ID
            
        Returns:
            改写后的问题
        """
        queue = self.session_queues.get(session_id)
        if not queue:
            return None
        
        # 获取当前队列中的所有参数对
        param_pairs = list(queue)
        
        # 调用意图改写API
        rewritten_questions = await self.intent_service.rewrite_intent(param_pairs)
        
        # 返回当前（最新）参数对对应的改写问题
        if rewritten_questions and len(rewritten_questions) > 0:
            return rewritten_questions[-1]  # 取最后一个（对应当前新增的参数对）
        
        return None
    
    async def _vector_search(self, query: str) -> Optional[str]:
        """
        向量检索候选答案
        
        Args:
            query: 查询问题
            
        Returns:
            候选答案
        """
        try:
            top_k = self.conv_config['vector_search_top_k']
            results = await self.vector_store.search_by_text(query, top_k)
            
            if results and len(results) > 0:
                # 取第一个结果作为候选答案，优先使用answer字段
                first_result = results[0]
                return first_result.get('answer', '') or first_result.get('text', '')
            
        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
        
        return None
    
    async def _polish_answer(self, question: str, candidate_answer: str) -> Optional[str]:
        """
        润色答案
        
        Args:
            question: 问题
            candidate_answer: 候选答案
            
        Returns:
            润色后的答案
        """
        try:
            polished = await self.polish_service.polish_answer(question, candidate_answer)
            return polished
        except Exception as e:
            self.logger.error(f"答案润色失败: {e}")
            return None
    
    def _build_result_record(
        self, 
        session_id: str, 
        service_name: str, 
        user_name: str, 
        time_str: str,
        question: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        构造结果记录
        
        Args:
            session_id: 会话ID
            service_name: 客服名称
            user_name: 用户名称
            time_str: 时间字符串
            question: 问题
            answer: 答案
            
        Returns:
            结果记录字典
        """
        fields = self.conv_config['csv_fields']
        
        return {
            fields['id_field']: session_id,
            fields['service_name_field']: service_name,
            fields['user_name_field']: user_name,
            fields['question_time_field']: time_str,
            fields['date_field']: datetime.now().strftime('%Y-%m-%d'),
            fields['question_field']: question,
            fields['answer_field']: answer,
            fields['image_url_field']: ''  # 暂留空
        }
    
    def _write_to_csv(self, records: List[Dict[str, Any]], output_file: str) -> None:
        """
        写入CSV文件
        
        Args:
            records: 记录列表
            output_file: 输出文件路径
        """
        if not records:
            self.logger.warning("没有记录需要写入")
            return
        
        fields = self.conv_config['csv_fields']
        fieldnames = list(fields.values())
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        
        self.logger.info(f"成功写入 {len(records)} 条记录到 {output_file}")