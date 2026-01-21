#!/bin/bash

echo "=== 多语种向量检索系统 ==="
echo ""

function show_help() {
    echo "使用方法:"
    echo "  ./run.sh build      - 构建Docker镜像"
    echo "  ./run.sh start      - 启动所有服务"
    echo "  ./run.sh pipeline   - 运行数据处理流程"
    echo "  ./run.sh service    - 启动检索服务"
    echo "  ./run.sh stop       - 停止所有服务"
    echo "  ./run.sh logs       - 查看日志"
    echo "  ./run.sh clean      - 清理容器和数据"
    echo ""
}

case "$1" in
    build)
        echo "构建Docker镜像..."
        cd docker && docker-compose build
        ;;
    start)
        echo "启动服务..."
        cd docker && docker-compose up -d etcd minio milvus-standalone
        echo "等待Milvus启动..."
        sleep 10
        ;;
    pipeline)
        echo "运行数据处理流程..."
        # cd docker && docker-compose run --rm app python3 -m src.pipeline
        cd docker && docker-compose run --rm app python3 -m src.core.pipeline
        ;;
    service)
        echo "启动检索服务..."
        # cd docker && docker-compose up -d service
        command: ["python3", "-m", "src.services.service"]
        echo "服务已启动: http://localhost:8000"
        ;;
    stop)
        echo "停止所有服务..."
        cd docker && docker-compose down
        ;;
    logs)
        cd docker && docker-compose logs -f $2
        ;;
    clean)
        echo "清理容器和数据..."
        cd docker && docker-compose down -v
        ;;
    *)
        show_help
        ;;
esac
