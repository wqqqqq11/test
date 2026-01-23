"""
意图改写服务模块
提供意图改写API调用功能（预留接口）
"""

import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple


class IntentService:
    """意图改写服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化意图改写服务
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.conv_config = config['conversation_processing']
        self.api_config = self.conv_config['api_endpoints']
        self.logger = logging.getLogger(__name__)
        
        self.intent_rewrite_url = self.api_config['intent_rewrite_url']
        self.timeout = 30  # 请求超时时间
        self.max_retries = 3  # 最大重试次数
    
    async def rewrite_intent(self, param_pairs: List[Tuple[str, str]]) -> Optional[List[str]]:
        """
        调用意图改写API
        
        Args:
            param_pairs: 参数对列表 [(question, answer), ...]
            
        Returns:
            改写后的问题列表，与输入参数对一一对应
        """
        if not param_pairs:
            return None
        
        try:
            # 构造请求数据
            request_data = {
                "param_pairs": [
                    {
                        "question": pair[0],
                        "answer": pair[1]
                    } for pair in param_pairs
                ],
                "context": {
                    "window_size": len(param_pairs),
                    "timestamp": self._get_current_timestamp()
                }
            }
            
            # 发送API请求
            response_data = await self._make_api_request(
                self.intent_rewrite_url, 
                request_data
            )
            
            if response_data and 'rewritten_questions' in response_data:
                return response_data['rewritten_questions']
            else:
                self.logger.warning("意图改写API返回格式异常")
                return None
                
        except Exception as e:
            self.logger.error(f"意图改写API调用失败: {e}")
            # 返回None，调用方会使用原问题作为fallback
            return None
    
    async def _make_api_request(self, url: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        发送API请求
        
        Args:
            url: API地址
            data: 请求数据
            
        Returns:
            响应数据
        """
        for attempt in range(self.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        url,
                        json=data,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            raise Exception(f"API返回状态码: {response.status}")
                            
            except Exception as e:
                self.logger.warning(f"API请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(1 * (attempt + 1))  # 指数退避
        
        raise Exception("API请求超过最大重试次数")
    
    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳
        
        Returns:
            时间戳字符串
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def is_service_available(self) -> bool:
        """
        检查服务是否可用
        
        Returns:
            服务是否可用
        """
        # TODO: 实现健康检查逻辑
        # 可以发送ping请求或检查服务状态
        return True
    
    async def health_check(self) -> bool:
        """
        异步健康检查
        
        Returns:
            服务是否健康
        """
        try:
            # 发送健康检查请求
            health_url = f"{self.intent_rewrite_url.rstrip('/')}/health"
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.debug(f"意图改写服务健康检查失败: {e}")
            return False