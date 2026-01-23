"""
LLM服务模块
提供语义判断、问题融合等LLM相关功能
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import openai
from openai import AsyncOpenAI


class LLMService:
    """LLM服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM服务
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.qwen_config = config['qwen']
        self.conv_config = config['conversation_processing']
        self.logger = logging.getLogger(__name__)
        
        # 初始化OpenAI客户端（兼容通义千问）
        self.client = AsyncOpenAI(
            api_key=self.qwen_config['api_key'],
            base_url=self.qwen_config['base_url']
        )
        
        self.model = self.qwen_config['model']
        self.max_retries = self.qwen_config['max_retries']
        self.timeout = self.qwen_config['timeout']
    
    async def validate_semantic_meaning(self, content: str) -> bool:
        """
        验证内容是否具有语义意义
        
        Args:
            content: 待验证的内容
            
        Returns:
            是否有意义
        """
        if not content or not content.strip():
            return False
        
        # 简单规则过滤
        content = content.strip()
        
        # 过滤纯符号、表情符号等
        if self._is_meaningless_content(content):
            return False
        
        # 使用LLM进行语义判断
        prompt = self.conv_config['llm_prompts']['semantic_validation'].format(
            content=content
        )
        
        try:
            response = await self._call_llm(prompt)
            return '有意义' in response
        except Exception as e:
            self.logger.error(f"语义验证失败: {e}")
            # 失败时采用保守策略，认为有意义
            return True
    
    async def check_relevance(self, question: str, answer: str) -> bool:
        """
        检查问题与答案的相关性
        
        Args:
            question: 客户问题
            answer: 客服答案
            
        Returns:
            是否相关
        """
        if not question or not answer:
            return False
        
        prompt = self.conv_config['llm_prompts']['relevance_check'].format(
            question=question.strip(),
            answer=answer.strip()
        )
        
        try:
            response = await self._call_llm(prompt)
            return '相关' in response
        except Exception as e:
            self.logger.error(f"相关性检查失败: {e}")
            # 失败时采用保守策略，认为相关
            return True
    
    async def fuse_questions(self, questions: List[str]) -> Optional[str]:
        """
        融合多个问题为一个标准化问题
        
        Args:
            questions: 问题列表
            
        Returns:
            融合后的问题
        """
        if not questions:
            return None
        
        if len(questions) == 1:
            return questions[0].strip()
        
        # 去重和清理
        unique_questions = []
        seen = set()
        for q in questions:
            q = q.strip()
            if q and q not in seen:
                unique_questions.append(q)
                seen.add(q)
        
        if len(unique_questions) == 1:
            return unique_questions[0]
        
        # 构造问题列表字符串
        questions_text = '\n'.join([f"{i+1}. {q}" for i, q in enumerate(unique_questions)])
        
        prompt = self.conv_config['llm_prompts']['question_fusion'].format(
            questions=questions_text
        )
        
        try:
            response = await self._call_llm(prompt)
            return response.strip() if response else None
        except Exception as e:
            self.logger.error(f"问题融合失败: {e}")
            # 失败时返回第一个问题
            return unique_questions[0] if unique_questions else None
    
    def _is_meaningless_content(self, content: str) -> bool:
        """
        检查内容是否无意义（基于规则）
        
        Args:
            content: 内容
            
        Returns:
            是否无意义
        """
        # 过短内容
        if len(content) < 2:
            return True
        
        # 纯符号或表情
        meaningless_patterns = [
            '#E-',  # 表情符号
            '嗯', '哦', '啊', '呃', '额',  # 语气词
            '...', '。。。', '~~~',  # 省略号
        ]
        
        for pattern in meaningless_patterns:
            if pattern in content:
                return True
        
        # 纯数字或字母
        if content.isdigit() or content.isalpha():
            return True
        
        # 长度过短且包含特殊字符
        if len(content) <= 5 and any(char in content for char in ['#', '@', '$', '%', '&']):
            return True
        
        return False
    
    async def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        调用LLM API
        
        Args:
            prompt: 提示词
            max_tokens: 最大token数
            
        Returns:
            LLM响应
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的对话数据分析助手。请严格按照要求回答，不要添加额外的解释或格式。"
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                    timeout=self.timeout
                )
                
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()
                else:
                    raise Exception("LLM返回空响应")
                    
            except Exception as e:
                self.logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(1 * (attempt + 1))  # 指数退避
        
        raise Exception("LLM调用超过最大重试次数")