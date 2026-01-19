# 多语种向量检索系统

基于聚类分析、自动标注和向量化的多语种数据处理系统，支持GPU加速和性能监控。

## 功能特性

- **聚类分析**: 使用MiniBatchKMeans对数据进行聚类
- **智能标注**: 通过Qwen大模型为每个聚类生成中文标签
- **向量化**: 使用CLIP多语言模型进行文本向量化
- **向量存储**: 将数据存储到Milvus向量数据库
- **检索服务**: 提供FastAPI接口进行向量检索
- **性能监控**: 实时监控GPU、内存使用并生成详细报表

## 系统架构

```
test/
├── outputs/              # 性能报表输出目录
├── config.json          # 统一配置文件
├── requirements.txt     # Python依赖
├── src/
│   ├── pipeline.py      # 主流程：聚类→标注→向量化→存储
│   ├── service.py       # FastAPI检索服务
│   ├── milvus_store.py  # Milvus数据库封装
│   ├── models.py        # CLIP和Qwen模型封装
│   ├── metrics.py       # 性能监控和报表生成
│   ├── io_utils.py      # CSV数据加载和处理
│   └── common.py        # 通用工具函数
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## 环境要求

- Docker & Docker Compose
- NVIDIA GPU (支持CUDA 11.8)
- NVIDIA Container Toolkit

## 快速开始

### 1. 准备数据

将CSV文件放置在 `data/input.csv`，确保包含以下列：
- `id`: 唯一标识符
- `title`: 标题
- `description`: 描述

### 2. 配置参数

编辑 `config.json` 调整参数：

```json
{
  "clustering": {
    "n_clusters": 10,
    "batch_size": 1000
  },
  "clip": {
    "model_name": "clip-ViT-B-32-multilingual-v1",
    "batch_size": 32
  }
}
```

### 3. 启动服务

```bash
cd docker
docker-compose up -d
```

### 4. 运行数据处理

```bash
运行这个
docker-compose run app python3 -m src.pipeline
```

### 5. 启动检索服务

```bash
docker-compose up service
```

## API使用

### 健康检查

```bash
curl http://localhost:8000/health
```

### 向量检索

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "搜索文本",
    "top_k": 10
  }'
```

响应格式：

```json
{
  "results": [
    {
      "id": "123",
      "cluster_id": 2,
      "cluster_label": "技术文档",
      "text": "文本内容...",
      "score": 0.95
    }
  ],
  "query": "搜索文本",
  "total": 10
}
```

## 性能报表

执行流程后，在 `outputs/` 目录下生成性能报表：

- `performance_report_YYYYMMDD_HHMMSS.json`: JSON格式详细数据
- `performance_report_YYYYMMDD_HHMMSS.html`: 可视化HTML报表

报表包含各阶段的：
- 执行时长
- GPU利用率
- 内存使用情况
- 时间占比分析

## 配置说明

### 数据配置

```json
"data": {
  "input_csv": "data/input.csv",
  "text_columns": ["title", "description"],
  "id_column": "id"
}
```

### 聚类配置

```json
"clustering": {
  "n_clusters": 10,
  "batch_size": 1000,
  "max_iter": 100
}
```

### Milvus配置

```json
"milvus": {
  "host": "milvus-standalone",
  "port": 19530,
  "collection_name": "multilingual_vectors",
  "dimension": 512
}
```

## 技术栈

- **机器学习**: scikit-learn, sentence-transformers
- **深度学习**: PyTorch, CLIP
- **向量数据库**: Milvus 2.3
- **Web框架**: FastAPI
- **监控**: psutil, nvidia-ml-py
- **容器化**: Docker, NVIDIA Container Toolkit

## 注意事项

1. 确保GPU驱动和CUDA版本匹配
2. 首次运行会下载CLIP模型（约1GB）
3. Milvus需要约2GB内存启动
4. 建议使用SSD存储以提升性能
5. Collection名称 `multilingual_vectors` 专用于本项目

## 故障排除

### GPU未识别

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Milvus连接失败

检查服务状态：
```bash
docker-compose ps
docker-compose logs milvus-standalone
```

### 内存不足

调整配置中的 `batch_size` 参数降低内存占用。

## 许可证

MIT License

