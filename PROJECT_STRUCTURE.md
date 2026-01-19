# 项目结构说明

```
test/
│
├── config.json                 # 核心配置文件，统一管理所有参数
├── requirements.txt            # Python依赖包列表
├── run.sh                      # 运行脚本（构建、启动、停止）
├── .gitignore                  # Git忽略规则
│
├── README.md                   # 项目说明文档
├── QUICKSTART.md               # 快速开始指南
│
├── data/                       # 数据目录
│   └── input.csv              # 输入CSV文件（示例）
│
├── outputs/                    # 输出目录
│   ├── .gitkeep               # 保留目录的占位文件
│   ├── performance_report_*.json   # 性能报表（JSON格式）
│   └── performance_report_*.html   # 性能报表（HTML格式）
│
├── src/                        # 源代码目录
│   ├── __init__.py            # 包初始化文件
│   │
│   ├── pipeline.py            # 主流程控制
│   │   ├── 数据加载
│   │   ├── 聚类分析（MiniBatchKMeans）
│   │   ├── 标签生成（Qwen）
│   │   ├── 向量化（CLIP）
│   │   └── 存储到Milvus
│   │
│   ├── service.py             # FastAPI检索服务
│   │   ├── /health - 健康检查
│   │   └── /query - 向量检索
│   │
│   ├── milvus_store.py        # Milvus数据库封装
│   │   ├── connect() - 连接数据库
│   │   ├── create_collection() - 创建集合
│   │   ├── insert() - 插入数据
│   │   └── search() - 向量检索
│   │
│   ├── models.py              # 模型封装
│   │   ├── CLIPEmbedder - CLIP向量化模型
│   │   └── QwenLabeler - Qwen标注模型
│   │
│   ├── metrics.py             # 性能监控
│   │   ├── track_stage() - 追踪各阶段性能
│   │   ├── _get_gpu_metrics() - GPU监控
│   │   └── generate_report() - 生成报表
│   │
│   ├── io_utils.py            # 数据IO工具
│   │   ├── load_csv() - 加载CSV
│   │   ├── extract_text() - 提取文本
│   │   └── prepare_data() - 数据准备
│   │
│   └── common.py              # 通用工具
│       ├── load_config() - 加载配置
│       ├── setup_logger() - 设置日志
│       ├── retry() - 重试装饰器
│       └── ensure_dir() - 确保目录存在
│
└── docker/                     # Docker相关文件
    ├── Dockerfile             # 应用镜像定义
    └── docker-compose.yml     # 服务编排配置
        ├── etcd - Milvus元数据存储
        ├── minio - Milvus对象存储
        ├── milvus-standalone - 向量数据库
        ├── app - 数据处理容器
        └── service - API服务容器
```

## 核心模块说明

### 1. pipeline.py - 主流程
负责协调整个数据处理流程，按顺序执行：
- 阶段1：加载CSV数据
- 阶段2：MiniBatchKMeans聚类
- 阶段3：通过Qwen生成聚类标签
- 阶段4：使用CLIP模型向量化
- 阶段5：存储到Milvus数据库

### 2. service.py - API服务
提供RESTful API接口：
- GET /health：服务健康检查
- POST /query：向量检索查询

### 3. milvus_store.py - 向量数据库
封装Milvus操作：
- 连接管理
- Collection创建（专用集合：multilingual_vectors）
- 批量插入
- 向量检索

### 4. models.py - 模型管理
封装两个核心模型：
- CLIPEmbedder：多语言文本向量化
- QwenLabeler：智能聚类标注

### 5. metrics.py - 性能监控
实时监控并记录：
- GPU利用率和显存
- CPU和内存使用
- 各阶段执行时间
- 生成JSON和HTML报表

### 6. io_utils.py - 数据处理
处理CSV文件：
- 读取和验证
- 字段映射
- 文本拼接

### 7. common.py - 工具函数
提供通用功能：
- 配置加载
- 日志设置
- 重试机制
- 目录管理

## 数据流转

```
CSV文件 
  ↓
DataLoader (io_utils.py)
  ↓
文本数据
  ↓
CLIPEmbedder (models.py) → 聚类 (MiniBatchKMeans)
  ↓                            ↓
向量                          聚类ID
  ↓                            ↓
QwenLabeler (models.py) ← 聚类样本
  ↓
中文标签
  ↓
MilvusStore (milvus_store.py)
  ↓
向量数据库
  ↓
FastAPI Service (service.py)
  ↓
用户查询结果
```

## 配置管理

所有参数在 `config.json` 中统一管理：

- **data**: 数据源配置
- **clustering**: 聚类参数
- **qwen**: Qwen API配置
- **clip**: CLIP模型配置
- **milvus**: 向量数据库配置
- **service**: API服务配置
- **performance**: 性能监控配置
- **logging**: 日志配置

## Docker架构

### 服务依赖关系
```
app/service (应用)
    ↓
milvus-standalone (向量数据库)
    ↓
├── etcd (元数据)
└── minio (对象存储)
```

### GPU配置
两个服务容器都配置了GPU支持：
- app：用于模型推理（CLIP、聚类）
- service：用于向量编码（查询时）

## 性能报表内容

### JSON格式
```json
{
  "timestamp": "20240118_143000",
  "stages": {
    "数据加载": {
      "duration_seconds": 2.5,
      "start_metrics": {...},
      "end_metrics": {...}
    }
  },
  "summary": {
    "total_duration_seconds": 150.3,
    "stages_breakdown": {...}
  }
}
```

### HTML格式
可视化展示：
- 总体概况
- 各阶段耗时表格
- GPU/内存使用情况
- 时间占比分析

## 扩展点

1. **自定义模型**：在 models.py 中添加新的Embedder
2. **数据源**：在 io_utils.py 中支持其他格式
3. **存储后端**：在 milvus_store.py 中切换其他向量库
4. **监控指标**：在 metrics.py 中添加自定义指标
5. **API接口**：在 service.py 中添加新的端点
