# 快速开始指南

## 环境准备

### 1. 安装必要工具

```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. 验证GPU支持

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 运行步骤

### 方式一：使用运行脚本（推荐）

```bash
# 1. 构建镜像
./run.sh build

# 2. 启动基础服务
./run.sh start

# 3. 运行数据处理流程
./run.sh pipeline

# 4. 启动检索服务
./run.sh service

# 5. 查看日志
./run.sh logs service

# 6. 停止服务
./run.sh stop
```

### 方式二：手动执行

```bash
# 1. 构建镜像
cd docker
docker-compose build

# 2. 启动Milvus和依赖
docker-compose up -d etcd minio milvus-standalone

# 3. 等待服务就绪（约10-15秒）
sleep 15

# 4. 运行数据处理
docker-compose run --rm app python3 -m src.pipeline

# 5. 启动检索服务
docker-compose up -d service
```

## 使用API

### 健康检查

```bash
curl http://localhost:8000/health
```

预期输出：
```json
{
  "status": "healthy",
  "service": "multilingual-vector-search"
}
```

### 向量检索

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "人工智能",
    "top_k": 5
  }'
```

预期输出：
```json
{
  "results": [
    {
      "id": "1",
      "cluster_id": 0,
      "cluster_label": "技术主题",
      "text": "人工智能技术发展 深度学习和神经网络...",
      "score": 0.95
    }
  ],
  "query": "人工智能",
  "total": 5
}
```

### Python客户端示例

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "云计算架构",
        "top_k": 10
    }
)

results = response.json()
for item in results['results']:
    print(f"ID: {item['id']}")
    print(f"标签: {item['cluster_label']}")
    print(f"相似度: {item['score']:.3f}")
    print(f"内容: {item['text'][:100]}...")
    print("-" * 50)
```

## 查看性能报表

处理完成后，报表保存在 `outputs/` 目录：

```bash
# 查看JSON报表
cat outputs/performance_report_*.json | jq .

# 用浏览器打开HTML报表
firefox outputs/performance_report_*.html
# 或
chromium outputs/performance_report_*.html
```

## 自定义数据

### 1. 准备CSV文件

将你的数据文件放在 `data/input.csv`，格式如下：

```csv
id,title,description
1,标题1,描述1
2,标题2,描述2
```

### 2. 修改配置

编辑 `config.json`：

```json
{
  "data": {
    "input_csv": "data/input.csv",
    "text_columns": ["title", "description"],
    "id_column": "id"
  },
  "clustering": {
    "n_clusters": 20  // 根据数据量调整
  }
}
```

### 3. 重新运行

```bash
./run.sh pipeline
```

## 常见问题

### Q: GPU内存不足

A: 减小批处理大小：
```json
{
  "clip": {
    "batch_size": 16  // 默认32，可降低到16或8
  },
  "clustering": {
    "batch_size": 500  // 默认1000
  }
}
```

### Q: Milvus连接失败

A: 检查服务状态：
```bash
docker-compose ps
docker-compose logs milvus-standalone
```

确保端口19530未被占用：
```bash
netstat -tuln | grep 19530
```

### Q: Qwen API调用失败

A: 检查API密钥和余额：
- 登录阿里云控制台
- 验证API_KEY是否有效
- 检查账户余额

### Q: 如何清理数据重新开始

A: 完全清理：
```bash
./run.sh clean
./run.sh start
./run.sh pipeline
```

## 性能优化建议

### 小数据集（< 10K条）
```json
{
  "clustering": {"n_clusters": 10, "batch_size": 500},
  "clip": {"batch_size": 32}
}
```

### 中等数据集（10K - 100K条）
```json
{
  "clustering": {"n_clusters": 50, "batch_size": 2000},
  "clip": {"batch_size": 64}
}
```

### 大数据集（> 100K条）
```json
{
  "clustering": {"n_clusters": 100, "batch_size": 5000},
  "clip": {"batch_size": 128}
}
```

## 监控和调试

### 实时查看GPU使用

```bash
watch -n 1 nvidia-smi
```

### 查看容器资源

```bash
docker stats
```

### 进入容器调试

```bash
docker-compose exec service bash
```

## 下一步

- 集成到你的应用
- 添加更多数据源
- 优化检索参数
- 扩展到多GPU训练
