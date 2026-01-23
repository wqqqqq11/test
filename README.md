# 多语种向量检索系统

基于文档处理、智能QA生成和向量化的多语种数据处理系统，支持GPU加速和性能监控。

## 功能特性

- **多格式文档处理**: 支持PDF、Word、Markdown、HTML、CSV等多种文档格式
- **智能文档清洗**: 自动去除页眉页脚、目录页，按标题结构合并内容
- **智能QA生成**: 使用Qwen大模型基于文档内容生成高质量问答对
- **QA质量校验**: 自动校验问答对的指代明确性和质量，确保可检索性
- **向量化**: 使用CLIP多语言模型进行文本向量化
- **向量存储**: 将数据存储到Milvus向量数据库
- **检索服务**: 提供FastAPI接口进行向量检索
- **性能监控**: 实时监控GPU、内存使用并生成详细报表

## 系统架构

```
test/
├── data/                # 数据存储目录
│   ├── .gitkeep
│   ├── raw_qa.csv       # 原始QA数据
│   └── processed_qa_pairs_*.csv  # 处理后的QA对
├── docker/              # Docker 部署配置
│   ├── docker-compose.yml
│   └── Dockerfile
├── MD文件/              # Markdown 文档目录
├── outputs/             # 输出文件目录（性能报表）
├── temp/                # 临时文件目录
├── src/                 # 核心源码目录
│   ├── core/            # 核心逻辑层
│   │   ├── document_processor.py  # 文档处理与QA生成
│   │   ├── qa_validator.py        # QA质量校验
│   │   └── pipeline.py            # 数据处理管道
│   ├── models/          # 数据模型层
│   │   └── models.py
│   ├── repositories/    # 数据存储层
│   │   └── milvus_store.py
│   ├── services/        # 业务服务层
│   │   └── service.py   # FastAPI服务
│   ├── utils/           # 工具模块
│   │   ├── common.py
│   │   ├── io_utils.py
│   │   └── metrics.py
│   ├── prompts.py       # 提示词配置
│   └── __init__.py
├── config.json          # 项目配置文件
├── requirements.txt     # Python 依赖清单
└── run.sh               # 项目启动脚本
```

## 文档处理流程

1. **文档加载**: 按页/段落读取为Document对象
2. **文本清洗**: 去除页眉页脚、目录页，规范化空白行
3. **结构合并**: 按标题/小节结构合并内容，保证跨页上下文连续
4. **文本分块**: 根据chunk_size和chunk_overlap进行智能切片
5. **QA生成**: 为每个chunk生成最多3个问答对
6. **质量校验**: 校验问答对的指代明确性，过滤不合格样本
7. **数据保存**: 将校验通过的QA对保存为CSV文件

## 环境要求

- Docker & Docker Compose
- NVIDIA GPU (支持CUDA 11.8)
- NVIDIA Container Toolkit

## 快速开始

### 1. 配置参数

编辑 `config.json` 调整参数：

```json
{
  "qwen": {
    "api_key": "your-api-key",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-flash",
    "timeout": 30
  },
  "document_processing": {
    "chunk_size": 1000,
    "chunk_overlap": 350,
    "max_qa_pairs_per_chunk": 3,
    "enable_qa_validation": true
  }
}
```

### 2. 启动服务

```bash
cd docker
docker-compose up -d
docker compose build service
```

### 3. 处理文档

#### 方式一：通过API上传文件

```bash
curl -X POST http://localhost:8000/process-uploaded-files \
  -F "files=@document.pdf" \
  -F "service_name=AI销售" \
  -F "user_name=客户A"
```

#### 方式二：处理服务器上的文件

```bash
curl -X POST http://localhost:8000/process-files \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["/path/to/document.pdf"],
    "service_name": "AI销售",
    "user_name": "客户A"
  }'
```

### 4. 启动检索服务

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
    "search_info": {
        "query": "你这手机多少钱",
        "timestamp": "2026-01-22T01:28:26.966896",
        "total_results": 2
    },
    "categories": [
        {
            "category_name": "自有定制",
            "items": [
                {
                    "similarity_score": 92.53,
                    "question": "这个手机价格多少钱？",
                    "answer": "哪一款呢亲爱哒~\n112800元的哦",
                    "image_url": "https://..."
                }
            ]
        }
    ]
}
```

### 处理文档文件

```bash
curl -X POST http://localhost:8000/process-uploaded-files \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "service_name=AI销售" \
  -F "user_name=客户A"
```

响应格式：

```json
{
    "success": true,
    "message": "成功处理2个文件",
    "total_qa_pairs": 45,
    "output_path": "data/processed_qa_pairs_20260123_080824.csv"
}
```

## 配置说明

### 文档处理配置

```json
"document_processing": {
  "chunk_size": 1000,              // 文本块大小（字符数）
  "chunk_overlap": 350,             // 文本块重叠大小
  "max_qa_pairs_per_chunk": 3,      // 每个chunk最多生成QA对数
  "max_content_length": 4000,       // 最大内容长度
  "temp_dir": "temp",               // 临时文件目录
  "supported_extensions": [         // 支持的文档格式
    ".txt", ".pdf", ".docx", ".doc", 
    ".md", ".html", ".csv", ".epub", 
    ".ppt", ".pptx", ".odt", ".eml", ".enex"
  ],
  "enable_qa_validation": true,    // 是否启用QA校验
  "validation_batch_size": 10       // 校验批次大小
}
```

### Qwen模型配置

```json
"qwen": {
  "api_key": "your-api-key",        // API密钥（可选，本地部署可不填）
  "base_url": "https://...",        // API基础URL
  "model": "qwen-flash",             // 模型名称
  "max_retries": 3,                  // 最大重试次数
  "timeout": 30                      // 超时时间（秒）
}
```

### 聚类配置

```json
"clustering": {
  "n_clusters": 30,                  // 聚类数量
  "batch_size": 5000,                // 批次大小
  "max_iter": 50,                    // 最大迭代次数
  "random_state": 42                  // 随机种子
}
```

### Milvus配置

```json
"milvus": {
  "host": "milvus-standalone",
  "port": 19530,
  "collection_name": "multilingual_vectors",
  "dimension": 384,                  // 向量维度
  "index_type": "IVF_FLAT",
  "metric_type": "COSINE",
  "nlist": 1024,
  "nprobe": 64
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

## 技术栈

- **文档处理**: langchain, PyMuPDF, unstructured
- **大语言模型**: OpenAI API (兼容Qwen)
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
6. QA校验使用qwen-flash模型，确保API可访问
7. 文档处理支持多种格式，但PDF处理效果最佳

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

### API调用失败

检查Qwen API配置是否正确，或使用本地部署的模型。

## 许可证

MIT License

git add .
git commit -m ""
git push -u origin master:main

git log --oneline

