import numpy as np
import json
import os
from typing import Dict, List, Any
from ..utils.common import load_config, setup_logger
from ..utils.io_utils import DataLoader
from ..models.models import CLIPEmbedder
from ..repositories.milvus_store import MilvusStore
from ..utils.metrics import MetricsCollector


class Pipeline:
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.logger = setup_logger("Pipeline", self.config)
        
        # 加载字段配置
        config_path_milvus = self.config.get('milvus', {}).get('milvus_config_path', 'tool_configs/milvus_config.json')
        if os.path.exists(config_path_milvus):
            with open(config_path_milvus, 'r', encoding='utf-8') as f:
                self.field_config = json.load(f)
        else:
            self.field_config = {"fields": [], "vector_field": {"name": "vector"}}
        
        self.fields_config = {field['name']: field for field in self.field_config.get('fields', [])}
        self.vector_field_name = self.field_config.get('vector_field', {}).get('name', 'vector')
        
        self.loader = DataLoader(self.config)
        self.embedder = CLIPEmbedder(self.config)
        self.store = MilvusStore(self.config)
        self.metrics = MetricsCollector(self.config)
    
    def run(self):
        self.logger.info("开始执行数据处理流程")
        
        with self.metrics.track_stage("数据加载"):
            self.logger.info("阶段1: 加载数据")
            df = self.loader.load_csv()
            records = self.loader.prepare_data(df)
            self.logger.info(f"加载了 {len(records)} 条记录")
        
        with self.metrics.track_stage("向量化"):
            self.logger.info("阶段2: 文本向量化")
            vectors = self._vectorize(records)
            self.logger.info(f"生成了 {len(vectors)} 个向量")
        
        with self.metrics.track_stage("数据存储"):
            self.logger.info("阶段3: 存储到Milvus")
            self._store_to_milvus(records, vectors)
            self.logger.info("数据存储完成")
        
        self.logger.info("生成性能报表")
        report = self.metrics.generate_report()
        self.logger.info(f"流程完成，总耗时: {report['summary']['total_duration_seconds']:.2f}秒")
        
        return report
    
    def _vectorize(self, records: List[Dict[str, Any]]) -> np.ndarray:
        texts = [str(r['raw'].get('question', '')) for r in records]
        return self.embedder.encode(texts)
    
    def _get_field_value(self, field_config: Dict[str, Any], record: Dict[str, Any], raw_data: Dict[str, Any]) -> Any:
        """根据字段配置获取值"""
        source_field = field_config.get('source_field', field_config['name'])
        default_value = field_config.get('default_value', '')
        max_length = field_config.get('max_source_length')
        
        # 优先从raw_data获取，其次从record获取
        if source_field in raw_data:
            value = raw_data[source_field]
        elif source_field in record:
            value = record[source_field]
        else:
            value = default_value
        
        # 转换为字符串并截断
        value_str = str(value) if value is not None else ''
        if max_length and len(value_str) > max_length:
            value_str = value_str[:max_length]
        
        # 根据字段类型转换
        dtype = field_config.get('dtype', 'VARCHAR')
        if dtype == 'INT64':
            try:
                return int(value_str) if value_str else int(default_value) if default_value else 0
            except (ValueError, TypeError):
                return int(default_value) if default_value else 0
        
        return value_str
    
    def _store_to_milvus(self, records: List[Dict[str, Any]], vectors: np.ndarray):
        self.store.connect()
        self.store.create_collection(drop_existing=True)
        
        data = []
        for i, record in enumerate(records):
            raw_data = record.get('raw', {})
            item = {}
            
            # 根据字段配置动态构建数据
            for field_config in self.field_config.get('fields', []):
                field_name = field_config['name']
                item[field_name] = self._get_field_value(field_config, record, raw_data)
            
            # 添加向量字段
            item[self.vector_field_name] = vectors[i]
            
            data.append(item)
        
        self.store.insert(data)
        self.store.load()
        self.logger.info(f"插入了 {len(data)} 条记录到Milvus")


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
