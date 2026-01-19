import json
import time
import psutil
import pynvml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import contextmanager


class MetricsCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config['performance']
        self.metrics = {}
        self.gpu_available = False
        
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            self.device_count = pynvml.nvmlDeviceGetCount()
        except:
            pass
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        if not self.gpu_available:
            return {}
        
        gpu_metrics = []
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            gpu_metrics.append({
                "device_id": i,
                "memory_used_mb": mem_info.used / 1024 / 1024,
                "memory_total_mb": mem_info.total / 1024 / 1024,
                "memory_percent": (mem_info.used / mem_info.total) * 100,
                "gpu_util_percent": util.gpu
            })
        
        return {"gpus": gpu_metrics}
    
    def _get_cpu_memory_metrics(self) -> Dict[str, Any]:
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024
        }
    
    @contextmanager
    def track_stage(self, stage_name: str):
        start_time = time.time()
        start_metrics = {**self._get_cpu_memory_metrics(), **self._get_gpu_metrics()}
        
        yield
        
        end_time = time.time()
        end_metrics = {**self._get_cpu_memory_metrics(), **self._get_gpu_metrics()}
        
        self.metrics[stage_name] = {
            "duration_seconds": end_time - start_time,
            "start_metrics": start_metrics,
            "end_metrics": end_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_report(self, output_dir: str = "outputs"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_data = {
            "timestamp": timestamp,
            "stages": self.metrics,
            "summary": self._generate_summary()
        }
        
        if "json" in self.config['report_format']:
            json_path = Path(output_dir) / f"performance_report_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        if "html" in self.config['report_format']:
            html_path = Path(output_dir) / f"performance_report_{timestamp}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_html(report_data))
        
        return report_data
    
    def _generate_summary(self) -> Dict[str, Any]:
        total_duration = sum(stage['duration_seconds'] for stage in self.metrics.values())
        
        summary = {
            "total_duration_seconds": total_duration,
            "stage_count": len(self.metrics),
            "stages_breakdown": {}
        }
        
        for stage_name, stage_data in self.metrics.items():
            summary['stages_breakdown'][stage_name] = {
                "duration_seconds": stage_data['duration_seconds'],
                "percentage": (stage_data['duration_seconds'] / total_duration * 100) if total_duration > 0 else 0
            }
        
        return summary
    
    def _generate_html(self, report_data: Dict[str, Any]) -> str:
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>性能报表 - {report_data['timestamp']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ color: #007bff; font-size: 1.1em; }}
        .stage-section {{ background: #f9f9f9; padding: 15px; margin: 15px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>性能分析报表</h1>
        <p>生成时间: {report_data['timestamp']}</p>
        
        <h2>总体概况</h2>
        <div class="metric">
            <span class="metric-label">总耗时:</span>
            <span class="metric-value">{report_data['summary']['total_duration_seconds']:.2f}秒</span>
        </div>
        <div class="metric">
            <span class="metric-label">阶段数:</span>
            <span class="metric-value">{report_data['summary']['stage_count']}</span>
        </div>
        
        <h2>各阶段详情</h2>
        <table>
            <tr>
                <th>阶段</th>
                <th>耗时(秒)</th>
                <th>占比(%)</th>
                <th>GPU利用率(%)</th>
                <th>内存使用率(%)</th>
            </tr>
"""
        
        for stage_name, stage_data in report_data['stages'].items():
            duration = stage_data['duration_seconds']
            percentage = report_data['summary']['stages_breakdown'][stage_name]['percentage']
            
            gpu_util = "N/A"
            if 'gpus' in stage_data['end_metrics'] and stage_data['end_metrics']['gpus']:
                gpu_util = f"{stage_data['end_metrics']['gpus'][0]['gpu_util_percent']:.1f}"
            
            mem_percent = stage_data['end_metrics'].get('memory_percent', 0)
            
            html += f"""
            <tr>
                <td>{stage_name}</td>
                <td>{duration:.2f}</td>
                <td>{percentage:.1f}</td>
                <td>{gpu_util}</td>
                <td>{mem_percent:.1f}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        return html
    
    def __del__(self):
        if self.gpu_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
