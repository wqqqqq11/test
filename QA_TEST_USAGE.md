# QA测试服务使用说明

## 概述

QA测试服务用于评估向量数据库中问答对的检索准确性。通过上传测试集，系统会使用测试集中的问题去查询向量数据库，并比较返回结果中的答案与测试集中的标准答案是否完全匹配。

## 功能特性

- 支持Recall@1、Recall@3、Recall@5指标计算
- 支持精确匹配验证（包括标点符号）
- 支持批量处理大规模测试数据
- 支持文件上传和本地文件测试
- 自动生成详细的测试报告

## 配置说明

### 测试配置文件 (`tool_configs/test_config.json`)

```json
{
  "test": {
    "recall_k_values": [1, 3, 5],           // 计算的Recall@K值
    "similarity_threshold": 0.0,            // 相似度阈值（暂未使用）
    "exact_match_required": true,           // 是否要求精确匹配
    "report_output_dir": "outputs/test_reports", // 报告输出目录
    "batch_size": 100                       // 批处理大小
  },
  "test_data": {
    "input_csv_path": "test_data/AI销售_测试1_20260125_155414.csv", // 默认测试文件路径
    "question_column": "question",          // 问题列名
    "answer_column": "answer",              // 答案列名
    "id_column": "ID"                       // ID列名
  },
  "vector_search": {
    "top_k": 10,                           // 向量搜索返回的Top-K结果数
    "search_timeout": 30                   // 搜索超时时间（秒）
  }
}
```

## API接口

### 1. 使用本地文件运行测试

**接口**: `POST /test/run-qa-test`

**请求体**:
```json
{
  "test_csv_path": "test_data/your_test_file.csv",  // 可选，默认使用配置文件中的路径
  "top_k": 10,                                      // 可选，默认使用配置文件中的值
  "recall_k_values": [1, 3, 5]                     // 可选，默认使用配置文件中的值
}
```

**响应**:
```json
{
  "success": true,
  "message": "测试完成，处理了 100 个查询",
  "metrics": {
    "recall_at_1": 0.8500,
    "recall_at_3": 0.9200,
    "recall_at_5": 0.9500,
    "total_queries": 100,
    "exact_matches": 85
  },
  "report_path": "outputs/test_reports/qa_test_report_20260125_160000.json",
  "timestamp": "2026-01-25T16:00:00"
}
```

### 2. 使用上传文件运行测试

**接口**: `POST /test/run-qa-test-upload`

**请求参数**:
- `file`: 上传的CSV测试文件
- `top_k`: 可选，Top-K值
- `recall_k_values`: 可选，逗号分隔的K值列表（如："1,3,5"）

**响应**: 同上

## 测试数据格式

测试CSV文件必须包含以下列：
- `ID`: 唯一标识符
- `question`: 测试问题
- `answer`: 标准答案
- 其他列可选

示例：
```csv
ID,service_name,user_name,question_time,data,question,answer,image_url,category
1,AI销售,客户A,2026-01-25 15:54:14,2026-01-25,有没有可能因为佩戴方式或皮肤状况导致无创血糖检测不准确？,是的，内置PPG生物追踪的光学传感器...,https://example.com/image.jpg,产品&功能咨询
```

## 指标说明

- **Recall@1**: 在Top-1结果中找到正确答案的查询比例
- **Recall@3**: 在Top-3结果中找到正确答案的查询比例  
- **Recall@5**: 在Top-5结果中找到正确答案的查询比例
- **Total Queries**: 总查询数量
- **Exact Matches**: 精确匹配的数量（等同于Recall@1的命中数）

## 使用步骤

1. **启动服务**
   ```bash
   python src/services/service.py
   ```

2. **准备测试数据**
   - 确保测试CSV文件格式正确
   - 将文件放置在指定位置或准备上传

3. **运行测试**
   - 使用API接口或测试脚本运行测试
   - 等待测试完成并查看结果

4. **查看报告**
   - 测试完成后会生成详细的JSON报告
   - 报告包含所有查询的详细结果和统计指标

## 测试脚本

可以使用提供的测试脚本验证服务：

```bash
python test_qa_service.py
```

该脚本会：
- 检查服务健康状态
- 运行基本QA测试
- 测试文件上传功能
- 显示测试结果

## 注意事项

1. **精确匹配**: 答案比较采用严格的字符串匹配，包括标点符号
2. **性能考虑**: 大量数据测试时建议调整batch_size参数
3. **超时设置**: 长时间测试需要适当调整timeout参数
4. **报告存储**: 测试报告会自动保存，注意磁盘空间
5. **向量数据库**: 确保Milvus服务正常运行且数据已加载

## 故障排除

1. **连接错误**: 检查Milvus服务是否正常运行
2. **文件格式错误**: 验证CSV文件格式和必要列是否存在
3. **内存不足**: 调整batch_size参数减少内存使用
4. **超时错误**: 增加search_timeout参数值