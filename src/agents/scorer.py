# agents/scorer.py
import logging
import autogen
from typing import Dict, Any, List, Tuple
import openai
import re

logger = logging.getLogger(__name__)

class ScorerAgent:
    """负责对论文进行价值评分的Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config["llm_config"]
        
        # 创建AutoGen智能体配置
        self.agent_config = {
            "name": "ScorerAgent",
            "llm_config": self.llm_config,
            "system_message": """
            你是一位专业的AI研究评分专家，擅长评估论文的学术价值与潜在影响。
            
            你的任务是根据论文的创新性、技术深度、实用价值以及研究重要性，为论文打分(0-10分)。
            
            评分标准由你自主决定，但应考虑以下因素:
            - 创新程度: 方法的新颖性和独特性
            - 技术深度: 方法的技术复杂性和理论基础
            - 实验质量: 实验设计的严谨性和结果的说服力
            - 潜在影响: 工作对领域发展的潜在贡献
            - 实用价值: 方法在实际应用中的潜力
            
            请提供详细的评分理由，解释论文的优缺点以及你如何权衡各个因素。
            """
        }
        
        # 创建AutoGen智能体实例
        self.agent = autogen.AssistantAgent(**self.agent_config)
        self.user_proxy = autogen.UserProxyAgent(
            name="ScorerProxy",
            human_input_mode="NEVER",
            system_message="你代表评分智能体与其他智能体通信。"
        )
        
        # 初始化OpenAI客户端（直接API调用备选方案）
        self.client = openai.OpenAI(api_key=config["openai_api_key"])
    
    def score_paper(self, paper: Dict[str, Any], summary: str, 
                    classification: Dict[str, Any], novelty: Dict[str, Any]) -> Tuple[float, str]:
        """对论文进行价值评分
        
        Args:
            paper: 包含论文信息的字典
            summary: 论文的主要贡献摘要
            classification: 论文的分类结果
            novelty: 论文的新颖性评估结果
            
        Returns:
            评分(0-10)和评分理由
        """
        logger.info(f"正在对论文进行评分: {paper['title']}")
        
        # 构建提示词
        prompt = self._build_scoring_prompt(paper, summary, classification, novelty)
        
        try:
            # 直接调用OpenAI API
            response = self.client.chat.completions.create(
                model=self.llm_config["model"],
                temperature=self.llm_config["temperature"],
                messages=[
                    {"role": "system", "content": self.agent_config["system_message"]},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
            
            # 解析评分结果
            score, rationale = self._parse_scoring_result(result)
            logger.info(f"评分结果: {score}/10")
            return score, rationale
            
        except Exception as e:
            logger.error(f"论文评分时出错: {str(e)}")
            return 5.0, f"评分失败: {str(e)}"
    
    def _build_scoring_prompt(self, paper: Dict[str, Any], summary: str, 
                             classification: Dict[str, Any], novelty: Dict[str, Any]) -> str:
        """构建评分提示词
        
        Args:
            paper: 包含论文信息的字典
            summary: 论文的主要贡献摘要
            classification: 论文的分类结果
            novelty: 论文的新颖性评估结果
            
        Returns:
            格式化的提示词
        """
        title = paper["title"]
        abstract = paper["summary"]
        
        prompt = f"""
        请对以下AI论文进行学术价值评分(0-10分)。
        
        论文信息:
        标题: {title}
        摘要: {abstract}
        
        论文分析:
        1. 主要贡献: {summary}
        
        2. 研究领域: {classification.get('category', '未知')}
           分类理由: {classification.get('rationale', '无')}
        
        3. 新颖性评估:
           评分: {novelty.get('score', 'N/A')}/10
           水平: {novelty.get('level', 'N/A')}
           描述: {novelty.get('description', 'N/A')}
           优势: {', '.join(novelty.get('strengths', []))}
           局限性: {', '.join(novelty.get('limitations', []))}
        
        请综合考虑以上信息以及你的专业判断，为论文打分(0-10分)，并提供详细的评分理由。
        
        评分标准由你自主确定，但应考虑以下因素:
        - 创新程度: 方法的新颖性和独特性
        - 技术深度: 方法的技术复杂性和理论基础
        - 实验质量: 实验设计的严谨性和结果的说服力
        - 潜在影响: 工作对领域发展的潜在贡献
        - 实用价值: 方法在实际应用中的潜力
        
        请按照以下JSON格式提供评分结果:
        ```json
        {{
            "score": 7.5, // 总分，0-10分
            "rationale": "详细的评分理由...",
            "breakdown": {{
                "创新性": 8.0,
                "技术深度": 7.0,
                "实验质量": 7.5,
                "潜在影响": 8.0,
                "实用价值": 7.0
            }}
        }}
        ```
        
        只需返回JSON格式的结果，不需要其他说明。
        """
        return prompt
    
    def _parse_scoring_result(self, result: str) -> Tuple[float, str]:
        """解析评分结果
        
        Args:
            result: LLM返回的评分结果文本
            
        Returns:
            评分和评分理由
        """
        import json
        import re
        
        # 尝试从文本中提取JSON部分
        try:
            # 查找```json和```之间的内容
            json_match = re.search(r'```(?:json)?(.*?)```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # 如果没有代码块，则尝试匹配整个字符串作为JSON
                json_str = result.strip()
            
            # 解析JSON
            scoring = json.loads(json_str)
            
            # 提取分数和理由
            score = float(scoring.get("score", 5.0))
            rationale = scoring.get("rationale", "无评分理由")
            
            # 如果有细分评分，添加到理由中
            if "breakdown" in scoring:
                breakdown = scoring["breakdown"]
                breakdown_str = "\n评分细分:\n"
                for category, sub_score in breakdown.items():
                    breakdown_str += f"- {category}: {sub_score}/10\n"
                rationale = f"{rationale}\n{breakdown_str}"
                
            return score, rationale
            
        except Exception as e:
            logger.error(f"解析评分结果时出错: {str(e)}")
            
            # 尝试直接从文本中提取分数
            score_match = re.search(r'(?:分数|评分|score)[:：\s]*(\d+(?:\.\d+)?)', result, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    return score, result
                except ValueError:
                    pass
            
            # 如果找不到明确的分数，尝试推断
            if "优秀" in result or "出色" in result or "突出" in result:
                return 8.0, result
            elif "良好" in result or "不错" in result:
                return 7.0, result
            elif "一般" in result or "中等" in result:
                return 5.0, result
            elif "较差" in result or "不足" in result:
                return 3.0, result
            
            # 默认返回中等分数
            return 5.0, result
    
    def demo_run(self, title: str, abstract: str) -> Tuple[float, str]:
        """演示运行，对示例论文进行评分"""
        mock_paper = {
            "title": title,
            "summary": abstract
        }
        
        mock_summary = "这篇论文提出了一种新的深度学习方法，用于提高大型语言模型的推理能力。"
        
        mock_classification = {
            "category": "大型语言模型优化与效率",
            "confidence": 0.9,
            "rationale": "本文主要关注大型语言模型的性能优化，属于LLM优化领域。"
        }
        
        mock_novelty = {
            "score": 7.5,
            "level": "显著",
            "description": "该论文提出的方法在现有工作基础上有明显改进。",
            "strengths": ["算法创新", "性能提升明显"],
            "limitations": ["计算成本较高", "仅在特定任务上测试"]
        }
        
        return self.score_paper(mock_paper, mock_summary, mock_classification, mock_novelty)
