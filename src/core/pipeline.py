import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Any
from ..utils.common import load_config, setup_logger
from ..utils.io_utils import DataLoader
from ..models.models import CLIPEmbedder, QwenLabeler
from ..repositories.milvus_store import MilvusStore
from ..utils.metrics import MetricsCollector


class Pipeline:
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.logger = setup_logger("Pipeline", self.config)
        
        self.loader = DataLoader(self.config)
        self.embedder = CLIPEmbedder(self.config)
        self.labeler = QwenLabeler(self.config)
        self.store = MilvusStore(self.config)
        self.metrics = MetricsCollector(self.config)
    
    def run(self):
        self.logger.info("开始执行数据处理流程")
        
        with self.metrics.track_stage("数据加载"):
            self.logger.info("阶段1: 加载数据")
            df = self.loader.load_csv()
            records = self.loader.prepare_data(df)
            self.logger.info(f"加载了 {len(records)} 条记录")
        
        with self.metrics.track_stage("聚类分析"):
            self.logger.info("阶段2: 聚类分析")
            clusters = self._clustering(records)
            self.logger.info(f"生成了 {len(set(clusters))} 个聚类")
        
        with self.metrics.track_stage("标签生成"):
            self.logger.info("阶段3: 生成聚类标签")
            cluster_labels = self._generate_labels(records, clusters)
            self.logger.info(f"生成了 {len(cluster_labels)} 个标签")
        
        with self.metrics.track_stage("向量化"):
            self.logger.info("阶段4: 文本向量化")
            vectors = self._vectorize(records)
            self.logger.info(f"生成了 {len(vectors)} 个向量")
        
        with self.metrics.track_stage("数据存储"):
            self.logger.info("阶段5: 存储到Milvus")
            self._store_to_milvus(records, clusters, cluster_labels, vectors)
            self.logger.info("数据存储完成")
        
        self.logger.info("生成性能报表")
        report = self.metrics.generate_report()
        self.logger.info(f"流程完成，总耗时: {report['summary']['total_duration_seconds']:.2f}秒")
        
        return report
    
    def _clustering(self, records: List[Dict[str, Any]]) -> np.ndarray:
        texts = [r['text'] for r in records]
        embeddings = self.embedder.encode(texts)
        
        kmeans = MiniBatchKMeans(
            n_clusters=self.config['clustering']['n_clusters'],
            batch_size=self.config['clustering']['batch_size'],
            max_iter=self.config['clustering']['max_iter'],
            random_state=self.config['clustering']['random_state']
        )
        
        clusters = kmeans.fit_predict(embeddings)
        return clusters
    
    def _generate_labels(self, records: List[Dict[str, Any]], clusters: np.ndarray) -> Dict[int, str]:
        labels = {}
        unique_clusters = set(clusters)
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(clusters == cluster_id)[0]
            samples = [records[i]['text'] for i in cluster_indices[:10]]
            label = self.labeler.generate_label(samples, cluster_id)
            labels[cluster_id] = label
            self.logger.info(f"聚类 {cluster_id}: {label}")
        
        return labels
    
    def _vectorize(self, records: List[Dict[str, Any]]) -> np.ndarray:
        texts = [r['text'] for r in records]
        return self.embedder.encode(texts)
    
    def _store_to_milvus(self, records: List[Dict[str, Any]], clusters: np.ndarray, 
                         cluster_labels: Dict[int, str], vectors: np.ndarray):
        self.store.connect()
        self.store.create_collection(drop_existing=True)
        
        data = []
        for i, record in enumerate(records):
            raw_data = record.get('raw', {})
            data.append({
                'id': str(record['id']),
                'cluster_id': int(clusters[i]),
                'cluster_label': cluster_labels[int(clusters[i])],
                'text': record['text'][:65000],
                'service_name': str(raw_data.get('service_name', ''))[:256],
                'user_name': str(raw_data.get('user_name', ''))[:256],
                'question_time': str(raw_data.get('question_time', ''))[:64],
                'data': str(raw_data.get('data', ''))[:64],
                'image_url': str(raw_data.get('image_url', ''))[:2048],
                'vector': vectors[i]
            })
        
        self.store.insert(data)
        self.store.load()
        self.logger.info(f"插入了 {len(data)} 条记录到Milvus")


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
