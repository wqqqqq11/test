from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple
from ..utils.common import load_config


class MilvusStore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config['milvus']
        self.collection = None
        
        # 加载字段配置
        config_path = self.config.get('milvus_config_path', 'tool_configs/milvus_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.field_config = json.load(f)
        else:
            # 默认配置
            self.field_config = {
                "fields": [],
                "vector_field": {"name": "vector", "dtype": "FLOAT_VECTOR"}
            }
        
        self.fields_config = {field['name']: field for field in self.field_config.get('fields', [])}
        self.vector_field_name = self.field_config.get('vector_field', {}).get('name', 'vector')
    
    def connect(self):
        connections.connect(
            alias="default",
            host=self.config['host'],
            port=self.config['port']
        )
    
    def _get_field_schema(self, field_config: Dict[str, Any]) -> FieldSchema:
        """根据配置创建字段schema"""
        dtype_map = {
            "VARCHAR": DataType.VARCHAR,
            "INT64": DataType.INT64,
            "INT32": DataType.INT32,
            "FLOAT": DataType.FLOAT,
            "DOUBLE": DataType.DOUBLE,
            "BOOL": DataType.BOOL,
            "FLOAT_VECTOR": DataType.FLOAT_VECTOR
        }
        
        dtype = dtype_map.get(field_config['dtype'], DataType.VARCHAR)
        kwargs = {"name": field_config['name'], "dtype": dtype}
        
        if dtype == DataType.VARCHAR:
            kwargs["max_length"] = field_config.get('max_length', 65535)
        
        if field_config.get('is_primary', False):
            kwargs["is_primary"] = True
        
        if dtype == DataType.FLOAT_VECTOR:
            kwargs["dim"] = self.config['dimension']
        
        return FieldSchema(**kwargs)
    
    def create_collection(self, drop_existing: bool = False):
        if drop_existing and utility.has_collection(self.config['collection_name']):
            try:
                utility.drop_collection(self.config['collection_name'])
            except Exception as e:
                # 忽略删除失败的错误，继续执行
                pass
        
        if utility.has_collection(self.config['collection_name']):
            self.collection = Collection(self.config['collection_name'])
            return
        
        # 根据配置动态创建字段
        fields = []
        for field_config in self.field_config.get('fields', []):
            fields.append(self._get_field_schema(field_config))
        
        # 添加向量字段
        vector_config = self.field_config.get('vector_field', {})
        if vector_config:
            vector_field = FieldSchema(
                name=vector_config['name'],
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config['dimension']
            )
            fields.append(vector_field)
        
        schema = CollectionSchema(fields=fields, description="Multilingual vectors")
        self.collection = Collection(name=self.config['collection_name'], schema=schema)
        
        index_params = {
            "index_type": self.config['index_type'],
            "metric_type": self.config['metric_type'],
            "params": {"nlist": self.config['nlist']}
        }
        self.collection.create_index(field_name=self.vector_field_name, index_params=index_params)
    
    def insert(self, data: List[Dict[str, Any]], batch_size: int = 1000):
        """根据字段配置动态插入数据"""
        # 获取所有字段名（按配置顺序）
        field_names = [field['name'] for field in self.field_config.get('fields', [])]
        field_names.append(self.vector_field_name)
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            entities = []
            
            # 按字段配置顺序构建数据
            for field_config in self.field_config.get('fields', []):
                field_name = field_config['name']
                default_value = field_config.get('default_value', '')
                entities.append([
                    item.get(field_name, default_value) for item in batch
                ])
            
            # 添加向量字段
            entities.append([
                item.get(self.vector_field_name).tolist() if hasattr(item.get(self.vector_field_name), 'tolist') 
                else item.get(self.vector_field_name) for item in batch
            ])
            
            self.collection.insert(entities)
        self.collection.flush()
    
    def load(self):
        self.collection.load()
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        nprobe = self.config.get('nprobe', 64)
        search_params = {"metric_type": self.config['metric_type'], "params": {"nprobe": nprobe}}
        
        # 获取所有需要输出的字段名
        output_fields = [field['name'] for field in self.field_config.get('fields', [])]
        
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field=self.vector_field_name,
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )
        
        output = []
        for hits in results:
            for hit in hits:
                result_item = {"score": float(hit.score)}
                # 动态获取所有字段
                for field_config in self.field_config.get('fields', []):
                    field_name = field_config['name']
                    default_value = field_config.get('default_value', '')
                    result_item[field_name] = hit.entity.get(field_name, default_value)
                output.append(result_item)
        
        return output
    
    def disconnect(self):
        connections.disconnect(alias="default")
