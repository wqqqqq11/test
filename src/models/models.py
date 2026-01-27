import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
from ..utils.common import retry


class CLIPEmbedder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config['clip']
        self.device = self.config['device'] if torch.cuda.is_available() else 'cpu'
        local_files_only = self.config.get('local_files_only', False)
        self.model = SentenceTransformer(
            self.config['model_name'], 
            device=self.device,
            local_files_only=local_files_only
        )
        self.batch_size = self.config['batch_size']
    
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_emb)
        return np.vstack(embeddings)
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class QwenLabeler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config['qwen']
        self.client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['base_url']
        )
    
    @retry(max_attempts=3, delay=2.0)
    def generate_label(self, samples: List[str], cluster_id: int) -> str:
        sample_text = '\n'.join([f"- {s[:200]}" for s in samples[:5]])
        
#         prompt = f"""以下是聚类 {cluster_id} 的样本文本，请为这个聚类生成一个简洁的中文标签（3-8个字）：

# {sample_text}

# 要求：
# 1. 标签要准确概括主题
# 2. 使用中文
# 3. 简洁明了

# 标签："""
        prompt = f"""以下是聚类 {cluster_id} 的样本文本，请为这个聚类生成一个简洁的中文标签：

        {sample_text}

        要求：
        1. 重要：固定只生成"功能咨询"和"品类咨询"两个标签！
        2. 使用中文

        标签："""
        
        response = self.client.chat.completions.create(
            model=self.config['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
