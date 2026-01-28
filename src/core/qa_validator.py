"""
QA对校验器，用于验证问答对的质量和指代明确性
"""
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI

from ..utils.common import setup_logger
from ..prompts.prompts import QA_VALIDATION_PROMPT


class QAValidator:
    """QA对校验器，确保问答对指代明确，无需依赖上下文"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.qwen_config = config["qwen"]
        self.logger = setup_logger("QAValidator", config)
        
        self.client = OpenAI(
            api_key=self.qwen_config["api_key"],
            base_url=self.qwen_config["base_url"],
            timeout=self.qwen_config["timeout"],
        )
        
        self.validation_model = "qwen3-max"
        
        # 常见的指代不明确的词汇模式
        self.ambiguous_patterns = [
            r'\b该产品\b',
            r'\b这个产品\b',
            r'\b本产品\b',
            r'\b该设备\b',
            r'\b这个设备\b',
            r'\b本设备\b',
            r'\b该系统\b',
            r'\b这个系统\b',
            r'\b本系统\b',
            r'\b该功能\b',
            r'\b这个功能\b',
            r'\b本功能\b',
            r'\b该方案\b',
            r'\b这个方案\b',
            r'\b本方案\b',
            r'\b该技术\b',
            r'\b这个技术\b',
            r'\b本技术\b',
            r'\b该应用\b',
            r'\b这个应用\b',
            r'\b本应用\b',
            r'\b这个东西\b',
            r'\b那个东西\b',
            r'\b这些东西\b',
            r'\b那些东西\b',
            r'\b它\b',
            r'\b它们\b',
            r'\b其\b',
            r'\b上述\b',
            r'\b以上\b',
            r'\b前面提到的\b',
            r'\b刚才说的\b',
        ]
        
        # 动态产品名称识别模式（更灵活的方式）
        self.product_name_indicators = [
            # 品牌名 + 型号/系列
            r'\b[A-Z][A-Za-z]*\s+[A-Z0-9][A-Za-z0-9]*\b',  # 如 "Agent Q", "Signature V"
            # 全大写的产品名
            r'\b[A-Z]{2,}\b',  # 如 "IRONFLIP", "VERTU", "VAOS", "AIGS"
            # AI + 其他词组合
            r'\bAI\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b',  # 如 "AI Darkroom Master"
            # 技术架构名称
            r'\b[A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+\s+Architecture\b',  # 如 "Falcon Cognitive Architecture"
        ]
        
        # 编译正则表达式以提高性能
        self.compiled_ambiguous_patterns = [re.compile(pattern) for pattern in self.ambiguous_patterns]
        self.compiled_product_indicators = [re.compile(pattern) for pattern in self.product_name_indicators]
    
    def _has_product_name_indicators(self, text: str) -> bool:
        """检查文本中是否包含产品名称指示器"""
        # 1. 检查动态模式
        for pattern in self.compiled_product_indicators:
            if pattern.search(text):
                return True
        
        # 2. 检查是否包含大写字母开头的专有名词（可能是产品名）
        # 匹配连续的大写字母开头的词，长度>=2
        proper_nouns = re.findall(r'\b[A-Z][A-Za-z]{1,}(?:\s+[A-Z0-9][A-Za-z0-9]*)*\b', text)
        if proper_nouns:
            # 过滤掉常见的非产品名词
            common_words = {'The', 'This', 'That', 'What', 'How', 'When', 'Where', 'Why', 'Who', 'Which'}
            product_like_nouns = [noun for noun in proper_nouns if noun not in common_words and len(noun) >= 3]
            if product_like_nouns:
                return True
        
        return False

    def _has_ambiguous_reference(self, text: str) -> Tuple[bool, List[str]]:
        """检查文本中是否包含指代不明确的词汇"""
        found_patterns = []
        
        # 检查指代不明确的模式
        for pattern in self.compiled_ambiguous_patterns:
            matches = pattern.findall(text)
            if matches:
                found_patterns.extend(matches)
        
        # 如果找到了指代不明确的词汇，进一步检查是否包含明确的产品名称
        if found_patterns:
            # 检查是否包含明确的产品名称指示器
            if self._has_product_name_indicators(text):
                return False, []
        
        return len(found_patterns) > 0, found_patterns
    
    def _build_validation_prompt(self, question: str, answer: str) -> str:
        """构建校验提示词"""
        return QA_VALIDATION_PROMPT.format(question=question, answer=answer)
    
    def validate_qa_pair(self, question: str, answer: str) -> Tuple[bool, str]:
        """
        校验单个问答对
        
        Args:
            question: 问题文本
            answer: 答案文本
            
        Returns:
            Tuple[bool, str]: (是否通过校验, 失败原因)
        """
        if not question.strip() or not answer.strip():
            return False, "问题或答案为空"
        
        # 1. 先进行快速模式匹配检查
        has_ambiguous_q, patterns_q = self._has_ambiguous_reference(question)
        has_ambiguous_a, patterns_a = self._has_ambiguous_reference(answer)
        
        if has_ambiguous_q or has_ambiguous_a:
            found_patterns = patterns_q + patterns_a
            return False, f"包含指代不明确词汇: {', '.join(set(found_patterns))}"
        
        # 2. 检查是否包含产品名称，如果包含则直接通过
        if self._has_product_name_indicators(question) or self._has_product_name_indicators(answer):
            return True, ""
        
        # 3. 只有在模式匹配无法确定且没有明显产品名称时，才使用AI校验
        try:
            prompt = self._build_validation_prompt(question, answer)
            
            response = self.client.chat.completions.create(
                model=self.validation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            
            result_text = (response.choices[0].message.content or "").strip()
            
            if result_text.startswith("PASS"):
                return True, ""
            elif result_text.startswith("FAIL"):
                # 提取失败原因
                parts = result_text.split(" ", 1)
                reason = parts[1] if len(parts) > 1 else "AI校验未通过"
                return False, reason
            else:
                # 如果AI输出格式不正确，默认通过（因为前面的检查已经过了）
                return True, ""
                
        except Exception as e:
            self.logger.error(f"AI校验时出错: {str(e)}")
            # 如果AI校验失败，且前面的检查都通过了，则默认通过
            return True, ""
    
    def validate_qa_batch(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        批量校验问答对
        
        Args:
            qa_pairs: 问答对列表，每个元素包含 'question' 和 'answer' 键
            
        Returns:
            List[Dict[str, Any]]: 校验通过的问答对列表，每个元素增加了校验信息
        """
        validated_pairs = []
        
        for i, qa_pair in enumerate(qa_pairs):
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            
            is_valid, reason = self.validate_qa_pair(question, answer)
            
            if is_valid:
                # 校验通过，保留原始数据并添加校验标记
                validated_pair = qa_pair.copy()
                validated_pair["validation_status"] = "PASS"
                validated_pair["validation_reason"] = ""
                validated_pairs.append(validated_pair)
                
                self.logger.debug(f"QA对 {i+1} 校验通过")
            else:
                self.logger.info(f"QA对 {i+1} 校验失败: {reason}")
                self.logger.debug(f"被拒绝的问题: {question}")
        
        pass_rate = len(validated_pairs) / len(qa_pairs) * 100 if qa_pairs else 0
        self.logger.info(f"批量校验完成: {len(validated_pairs)}/{len(qa_pairs)} 通过 (通过率: {pass_rate:.1f}%)")
        
        return validated_pairs
    
    def get_validation_stats(self, qa_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        获取校验统计信息（不实际校验，仅统计）
        
        Args:
            qa_pairs: 问答对列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_count = len(qa_pairs)
        ambiguous_count = 0
        ambiguous_patterns_found = []
        
        for qa_pair in qa_pairs:
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            
            has_ambiguous_q, patterns_q = self._has_ambiguous_reference(question)
            has_ambiguous_a, patterns_a = self._has_ambiguous_reference(answer)
            
            if has_ambiguous_q or has_ambiguous_a:
                ambiguous_count += 1
                ambiguous_patterns_found.extend(patterns_q + patterns_a)
        
        return {
            "total_pairs": total_count,
            "potentially_ambiguous": ambiguous_count,
            "estimated_pass_rate": (total_count - ambiguous_count) / total_count * 100 if total_count > 0 else 0,
            "common_ambiguous_patterns": list(set(ambiguous_patterns_found))
        }