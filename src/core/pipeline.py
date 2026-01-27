import json
import os
from typing import Dict, List, Any

import numpy as np

from ..utils.common import load_config, setup_logger
from ..utils.io_utils import DataLoader
from ..models.models import CLIPEmbedder
from ..repositories.milvus_store import MilvusStore
from ..utils.metrics import MetricsCollector


class Pipeline:
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.logger = setup_logger("Pipeline", self.config)

        self.field_config = self._load_field_config()
        self.fields = self.field_config.get("fields", [])
        self.vector_field_name = self.field_config.get("vector_field", {}).get("name", "vector")

        self.loader = DataLoader(self.config)
        self.embedder = CLIPEmbedder(self.config)
        self.store = MilvusStore(self.config)
        self.metrics = MetricsCollector(self.config)

    def _load_field_config(self) -> Dict[str, Any]:
        p = self.config.get("milvus", {}).get("milvus_config_path", "tool_configs/milvus_config.json")
        if not os.path.exists(p):
            return {"fields": [], "vector_field": {"name": "vector"}}
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("fields", [])
        cfg.setdefault("vector_field", {"name": "vector"})
        return cfg

    def run(self):
        csv_path = self.config.get("data", {}).get("input_csv", "")
        drop_existing = self.config.get("vectorization", {}).get("drop_existing", False)
        return self.vectorize_dataset(csv_path, drop_existing)

    def vectorize_dataset(self, csv_path: str, drop_existing: bool = False) -> Dict[str, Any]:
        self.logger.info(f"开始向量化数据集: {csv_path}")

        with self.metrics.track_stage("数据加载"):
            df = self.loader.load_csv(csv_path)
            records = self.loader.prepare_data(df)
            self.logger.info(f"加载了 {len(records)} 条记录")

        with self.metrics.track_stage("向量化"):
            vectors = self._vectorize(records)
            self.logger.info(f"生成了 {len(vectors)} 个向量")

        with self.metrics.track_stage("数据存储"):
            self._store_to_milvus(records, vectors, drop_existing)
            self.logger.info("数据存储完成")

        report_dir = self.config.get("vectorization", {}).get("report_output_dir", "outputs/vectorization_reports")
        report = self.metrics.generate_report(report_dir)
        self.logger.info(f"流程完成，总耗时: {report['summary']['total_duration_seconds']:.2f}秒")

        return {
            "success": True,
            "total_records": len(records),
            "duration_seconds": report["summary"]["total_duration_seconds"],
            "report": report,
        }

    def _vectorize(self, records: List[Dict[str, Any]]) -> np.ndarray:
        texts = [str((r.get("raw") or {}).get("question", "")) for r in records]
        return self.embedder.encode(texts)

    def _get_field_value(self, fc: Dict[str, Any], record: Dict[str, Any], raw: Dict[str, Any]) -> Any:
        name = fc["name"]
        source = fc.get("source_field") or name
        default = fc.get("default_value", "")

        value = raw.get(source, record.get(source, default))

        if fc.get("dtype") == "INT64":
            if value is None or value == "":
                return int(default) if str(default).strip() else 0
            try:
                return int(value)
            except (ValueError, TypeError):
                try:
                    return int(str(value).strip())
                except (ValueError, TypeError):
                    return int(default) if str(default).strip() else 0

        s = "" if value is None else str(value)
        limit = fc.get("max_source_length") or fc.get("max_length")
        if limit and len(s) > limit:
            s = s[:limit]
        return s

    def _validate_required(self, item: Dict[str, Any]):
        for fc in self.fields:
            if not fc.get("required"):
                continue
            k = fc["name"]
            v = item.get(k)
            if v is None or (isinstance(v, str) and v == ""):
                raise ValueError(f"required field missing: {k}")

    def _store_to_milvus(self, records: List[Dict[str, Any]], vectors: np.ndarray, drop_existing: bool = False):
        self.store.connect()
        self.store.create_collection(drop_existing=drop_existing)

        data = []
        for record, vec in zip(records, vectors):
            raw = record.get("raw") or {}

            item = {fc["name"]: self._get_field_value(fc, record, raw) for fc in self.fields}
            self._validate_required(item)

            item[self.vector_field_name] = vec
            data.append(item)

        self.store.insert(data)
        self.store.load()
        self.logger.info(f"插入了 {len(data)} 条记录到Milvus")


if __name__ == "__main__":
    Pipeline().run()
