"""
对话数据处理主程序
执行完整的客服对话数据处理流程
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.conversation_processor import ConversationProcessor
from src.utils.common import setup_logging


async def main():
    """主函数"""
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        config_path = project_root / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info("开始执行对话数据处理流程")
        logger.info(f"配置文件: {config_path}")
        
        # 检查输入文件是否存在
        input_file = config['conversation_processing']['input_file']
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            return
        
        # 创建输出目录
        output_file = config['conversation_processing']['output_csv']
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
        
        # 初始化处理器
        processor = ConversationProcessor(config)
        
        # 初始化向量数据库连接（如果需要）
        try:
            processor.vector_store.connect()
            processor.vector_store.load()
            logger.info("向量数据库连接成功")
        except Exception as e:
            logger.warning(f"向量数据库连接失败，将跳过向量检索步骤: {e}")
        
        # 执行处理流程
        await processor.process_conversations()
        
        logger.info("对话数据处理流程执行完成")
        
    except Exception as e:
        logger.error(f"处理流程执行失败: {e}", exc_info=True)
        raise
    
    finally:
        # 清理资源
        try:
            if 'processor' in locals():
                processor.vector_store.disconnect()
                logger.info("资源清理完成")
        except Exception as e:
            logger.warning(f"资源清理失败: {e}")


def run_sync():
    """同步运行入口"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_sync()