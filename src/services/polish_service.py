import httpx
import asyncio
from typing import Dict, List, Any
from ..utils.common import setup_logger


class PolishService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config['polish']
        self.logger = setup_logger("PolishService", config)
        base_url = self.config['base_url']
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"
        self.base_url = base_url.rstrip('/')
        self.timeout = self.config.get('timeout', 60)
        self.batch_size = self.config.get('batch_size', 50)
    
    async def _polish_batch(self, client: httpx.AsyncClient, batch: List[Dict[str, Any]]) -> List[str]:
        url = f"{self.base_url}/api/v1/answer/enhance"
        
        request_data = []
        for qa in batch:
            question = str(qa.get('question', '')).strip()
            answer = str(qa.get('answer', '')).strip()
            if question and answer:
                request_data.append({
                    "question": question,
                    "answer": answer
                })
        
        if not request_data:
            return []
        
        response = await client.post(url, json=request_data)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('code') == 200 and result.get('data'):
            enhanced_answers = result['data'].get('enhanced_answers', [])
            return enhanced_answers
        
        return []
    
    async def polish_qa_pairs(self, qa_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        polished_data = []
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            total_batches = (len(qa_data) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(0, len(qa_data), self.batch_size):
                batch = qa_data[batch_idx:batch_idx + self.batch_size]
                
                enhanced_answers = await self._polish_batch(client, batch)
                
                for idx, qa in enumerate(batch):
                    polished_qa = qa.copy()
                    if idx < len(enhanced_answers):
                        polished_qa['answer'] = enhanced_answers[idx]
                    polished_data.append(polished_qa)
                
                current_count = min(batch_idx + self.batch_size, len(qa_data))
                self.logger.info(f"已润色 {current_count}/{len(qa_data)} 条QA对")
        
        self.logger.info(f"润色完成，共处理 {len(polished_data)} 条QA对")
        return polished_data