"""
回答润色服务模块
提供回答润色API调用功能（预留接口）
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

import aiohttp


class PolishService:
    """回答润色服务类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化回答润色服务

        Args:
            config: 配置字典
        """
        self.config = config
        self.conv_config = config.get("conversation_processing", {})
        self.api_config = self.conv_config.get("api_endpoints", {})
        self.logger = logging.getLogger(__name__)

        # 适配实际接口：/api/v1/answer/enhance
        self.answer_polish_url = self.api_config.get("answer_polish_url", "").strip()
        if not self.answer_polish_url:
            raise ValueError("config.conversation_processing.api_endpoints.answer_polish_url is required")

        self.timeout = int(self.api_config.get("answer_polish_timeout", 30))
        self.max_retries = int(self.api_config.get("answer_polish_max_retries", 3))

    async def polish_answer(self, question: str, candidate_answer: str) -> Optional[str]:
        """
        调用回答润色API

        Args:
            question: 问题
            candidate_answer: 候选答案

        Returns:
            润色后的答案，失败返回 None（调用方可使用候选答案兜底）
        """
        question = (question or "").strip()
        candidate_answer = (candidate_answer or "").strip()
        if not question or not candidate_answer:
            return None

        request_data = {
            "question": question,
            "answer": candidate_answer
        }

        try:
            response_data = await self._make_api_request(self.answer_polish_url, request_data)
            enhanced = self._extract_enhanced_answer(response_data)
            if enhanced:
                return enhanced
            self.logger.warning("回答润色API返回格式异常或缺少 enhanced_answer 字段")
            return None
        except Exception as e:
            self.logger.error(f"回答润色API调用失败: {e}")
            return None

    async def batch_polish_answers(self, qa_pairs: List[Tuple[str, str]]) -> Optional[List[Optional[str]]]:
        """
        批量润色答案（当前接口未提供批量能力，这里采用并发逐条调用）

        Args:
            qa_pairs: 问答对列表 [(question, answer), ...]

        Returns:
            润色后的答案列表（与输入长度一致，单条失败返回 None）
        """
        if not qa_pairs:
            return None

        tasks = [
            self.polish_answer(q, a)
            for (q, a) in qa_pairs
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _make_api_request(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        发送API请求

        Args:
            url: API地址
            data: 请求数据

        Returns:
            响应数据（JSON）
        """
        last_err: Optional[Exception] = None

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {"Content-Type": "application/json"}

        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=data, headers=headers) as response:
                        # 200 才解析 json；非 200 尽量读取文本用于日志
                        if response.status != 200:
                            body = await self._safe_read_text(response)
                            raise RuntimeError(f"API返回状态码: {response.status}, body={body}")

                        return await response.json()

            except Exception as e:
                last_err = e
                self.logger.warning(f"API请求失败 (尝试 {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(attempt)  # 简单退避
                else:
                    break

        raise last_err if last_err else RuntimeError("API请求失败")

    @staticmethod
    def _extract_enhanced_answer(resp: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        解析接口返回的 enhanced_answer

        返回示例：
        {
          "code": 200,
          "message": "success",
          "data": {
            "enhanced_answer": "..."
          }
        }
        """
        if not isinstance(resp, dict):
            return None

        # 兼容：code 可能是 int/str
        code = resp.get("code", None)
        if code not in (200, "200"):
            return None

        data = resp.get("data")
        if not isinstance(data, dict):
            return None

        enhanced = data.get("enhanced_answer")
        if isinstance(enhanced, str) and enhanced.strip():
            return enhanced.strip()

        return None

    @staticmethod
    async def _safe_read_text(response: aiohttp.ClientResponse) -> str:
        try:
            return (await response.text())[:2000]
        except Exception:
            return ""

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳字符串"""
        return datetime.now().isoformat()

    def is_service_available(self) -> bool:
        """检查服务是否可用（占位）"""
        return True

    async def health_check(self) -> bool:
        """
        异步健康检查（按常见约定拼接 /health；若实际无该接口可关闭或改为配置）
        """
        try:
            health_url = f"{self.answer_polish_url.rstrip('/')}/health"
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.debug(f"回答润色服务健康检查失败: {e}")
            return False

    def get_polish_statistics(self) -> Dict[str, Any]:
        """获取润色统计信息（占位）"""
        return {
            "total_requests": 0,
            "success_rate": 0.0,
            "average_response_time": 0.0,
            "error_count": 0
        }
