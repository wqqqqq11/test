from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from typing import Dict, List, Any, Tuple


class MilvusStore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config['milvus']
        self.collection = None
    
    def connect(self):
        connections.connect(
            alias="default",
            host=self.config['host'],
            port=self.config['port']
        )
    
    def create_collection(self, drop_existing: bool = False):
        if drop_existing and utility.has_collection(self.config['collection_name']):
            utility.drop_collection(self.config['collection_name'])
        
        if utility.has_collection(self.config['collection_name']):
            self.collection = Collection(self.config['collection_name'])
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="cluster_id", dtype=DataType.INT64),
            FieldSchema(name="cluster_label", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=32768),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=32768),
            FieldSchema(name="service_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="user_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="question_time", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config['dimension'])
        ]
        
        schema = CollectionSchema(fields=fields, description="Multilingual vectors")
        self.collection = Collection(name=self.config['collection_name'], schema=schema)
        
        index_params = {
            "index_type": self.config['index_type'],
            "metric_type": self.config['metric_type'],
            "params": {"nlist": self.config['nlist']}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
    
    def insert(self, data: List[Dict[str, Any]], batch_size: int = 1000):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            entities = [
                [item['id'] for item in batch],
                [item['cluster_id'] for item in batch],
                [item['cluster_label'] for item in batch],
                [item['text'] for item in batch],
                [item.get('question', '') for item in batch],
                [item.get('answer', '') for item in batch],
                [item.get('service_name', '') for item in batch],
                [item.get('user_name', '') for item in batch],
                [item.get('question_time', '') for item in batch],
                [item.get('data', '') for item in batch],
                [item.get('image_url', '') for item in batch],
                [item.get('category', '') for item in batch],
                [item['vector'].tolist() for item in batch]
            ]
            self.collection.insert(entities)
        self.collection.flush()
    
    def load(self):
        self.collection.load()
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        nprobe = self.config.get('nprobe', 64)
        search_params = {"metric_type": self.config['metric_type'], "params": {"nprobe": nprobe}}
        
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["id", "cluster_id", "cluster_label", "text", "question", "answer", "service_name", "user_name", "question_time", "data", "image_url"]
        )
        
        output = []
        for hits in results:
            for hit in hits:
                output.append({
                    "id": hit.entity.get('id'),
                    "cluster_id": hit.entity.get('cluster_id'),
                    "cluster_label": hit.entity.get('cluster_label'),
                    "text": hit.entity.get('text'),
                    "question": hit.entity.get('question', ''),
                    "answer": hit.entity.get('answer', ''),
                    "service_name": hit.entity.get('service_name', ''),
                    "user_name": hit.entity.get('user_name', ''),
                    "question_time": hit.entity.get('question_time', ''),
                    "data": hit.entity.get('data', ''),
                    "image_url": hit.entity.get('image_url', ''),
                    "category": hit.entity.get('category', ''),
                    "score": float(hit.score)
                })
        
        return output
    
    def disconnect(self):
        connections.disconnect(alias="default")
