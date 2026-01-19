# 服务器部署指南

## 模型缓存优化

为了避免每次运行都重新下载模型（约1GB），项目已配置模型缓存持久化。

### 工作原理

1. **Docker Volume 持久化**：模型文件保存在 Docker volume `model_cache` 中
2. **环境变量配置**：明确指定缓存路径为 `/root/.cache`
3. **跨容器共享**：`app` 和 `service` 服务共享同一个模型缓存

### 首次运行

首次运行时，模型会被下载并保存到 volume 中：

```bash
docker-compose run --rm app python3 -m src.pipeline
```

下载完成后，模型会保存在 `model_cache` volume 中。

### 后续运行

后续运行时，模型会直接从缓存加载，**不会重新下载**：

```bash
# 即使删除容器，模型缓存仍然保留
docker-compose run --rm app python3 -m src.pipeline
```

### 查看模型缓存

```bash
# 查看所有 volumes
docker volume ls

# 查看 model_cache volume 的详细信息
docker volume inspect docker_model_cache

# 查看模型缓存大小（Linux/Mac）
docker run --rm -v docker_model_cache:/cache alpine sh -c "du -sh /cache"
```

### 清理模型缓存

如果需要重新下载模型（例如更新模型版本）：

```bash
# 停止所有服务
docker-compose down

# 删除模型缓存 volume
docker volume rm docker_model_cache

# 重新运行（会重新下载）
docker-compose run --rm app python3 -m src.pipeline
```

## 服务器部署步骤

### 1. 准备服务器环境

```bash
# 安装 Docker 和 Docker Compose
# 安装 NVIDIA Container Toolkit（如果使用GPU）

# 克隆项目
git clone <your-repo-url>
cd test
```

### 2. 配置环境

```bash
# 编辑配置文件
vim config.json

# 确保数据文件存在
ls data/input.csv
```

### 3. 首次部署

```bash
cd docker

# 启动基础设施服务
docker-compose up -d etcd minio milvus-standalone

# 等待 Milvus 就绪（约30-60秒）
sleep 60

# 运行 pipeline（首次会下载模型，约1GB，需要几分钟）
docker-compose run --rm app python3 -m src.pipeline

# 启动 API 服务
docker-compose up -d service
```

### 4. 验证部署

```bash
# 检查服务状态
docker-compose ps

# 检查 API 服务
curl http://localhost:8000/health

# 查看日志
docker-compose logs -f service
```

### 5. 后续更新

如果代码有更新，只需要：

```bash
# 拉取最新代码
git pull

# 重新构建镜像（如果需要）
docker-compose build app service

# 重启服务
docker-compose up -d --force-recreate service
```

**注意**：模型缓存会保留，不需要重新下载。

## 性能优化建议

### 1. 使用固定镜像标签

在生产环境中，建议使用固定版本的镜像标签，而不是 `latest`：

```yaml
# docker-compose.yml
services:
  app:
    image: your-registry/multilingual-pipeline:v1.0.0  # 固定版本
```

### 2. 预构建镜像

在 CI/CD 中预构建镜像并推送到镜像仓库：

```bash
# 构建并推送
docker build -t your-registry/multilingual-pipeline:v1.0.0 -f docker/Dockerfile .
docker push your-registry/multilingual-pipeline:v1.0.0
```

### 3. 使用镜像而不是构建

服务器上直接使用预构建的镜像：

```yaml
# docker-compose.yml
services:
  app:
    image: your-registry/multilingual-pipeline:v1.0.0
    # 移除 build 配置
```

### 4. 备份模型缓存

定期备份模型缓存 volume：

```bash
# 备份
docker run --rm \
  -v docker_model_cache:/source \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/model_cache_$(date +%Y%m%d).tar.gz -C /source .

# 恢复
docker run --rm \
  -v docker_model_cache:/target \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/model_cache_YYYYMMDD.tar.gz -C /target
```

## 故障排除

### 模型下载失败

如果模型下载失败（网络问题）：

```bash
# 检查网络连接
docker-compose run --rm app ping huggingface.co

# 使用代理（如果需要）
# 在 docker-compose.yml 中添加：
environment:
  - HTTP_PROXY=http://proxy:port
  - HTTPS_PROXY=http://proxy:port
```

### 磁盘空间不足

模型缓存约占用 1-2GB 空间：

```bash
# 检查磁盘使用
df -h

# 清理未使用的 volumes
docker volume prune
```

### 缓存不生效

如果模型仍然重新下载：

```bash
# 检查 volume 是否正确挂载
docker-compose run --rm app ls -la /root/.cache

# 检查环境变量
docker-compose run --rm app env | grep CACHE
```

## 监控和维护

### 定期检查

```bash
# 检查服务健康状态
docker-compose ps

# 查看资源使用
docker stats

# 查看日志
docker-compose logs --tail=100 service
```

### 日志轮转

配置日志轮转避免日志文件过大：

```yaml
# docker-compose.yml
services:
  service:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 安全建议

1. **不要提交敏感信息**：确保 `config.json` 中的 API key 等敏感信息不要提交到 Git
2. **使用环境变量**：生产环境使用环境变量或密钥管理服务
3. **限制网络访问**：使用防火墙限制不必要的端口访问
4. **定期更新**：定期更新基础镜像和安全补丁
