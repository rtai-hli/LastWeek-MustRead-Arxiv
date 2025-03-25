# agents/novelty_assessor.py
import logging
import autogen
from typing import Dict, Any, List
import openai
import re

logger = logging.getLogger(__name__)

class NoveltyAssessorAgent:
    """负责评估论文新颖性和增量贡献的Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config["llm_config"]
        
        # 创建AutoGen智能体配置
        self.agent_config = {
            "name": "NoveltyAssessorAgent",
            "llm_config": self.llm_config,
            "system_message": """
            你是一位专业的AI研究评审专家，擅长评估论文的新颖性和增量贡献。
            你需要分析论文内容，特别是引言、相关工作和方法部分，判断论文的创新程度。
            
            评估时请考虑以下几点:
            1. 论文是提出全新方法还是对已有方法的改进?
            2. 相对于现有工作，改进的幅度有多大?
            3. 论文的技术思路是否有独特之处?
            4. 论文解决的问题难度和重要性如何?
            
            请寻找论文中如"首次提出"、"突破性工作"、"显著超越"等表示新颖性的描述，
            也要注意论文是否客观描述了自身的局限性。
            
            为每篇论文提供新颖性评分(1-10分)和详细的评估理由。
            """
        }
        
        # 创建AutoGen智能体实例
        self.agent = autogen.AssistantAgent(**self.agent_config)
        self.user_proxy = autogen.UserProxyAgent(
            name="NoveltyAssessorProxy",
            human_input_mode="NEVER",
            system_message="你代表新颖性评估智能体与其他智能体通信。"
        )
        
        # 初始化OpenAI客户端（直接API调用备选方案）
        self.client = openai.OpenAI(api_key=config["openai_api_key"])
    
    def assess_novelty(self, paper: Dict[str, Any], summary: str) -> Dict[str, Any]:
        """评估论文的新颖性和增量贡献
        
        Args:
            paper: 包含论文信息的字典
            summary: 论文的主要贡献摘要（由摘要Agent生成）
            
        Returns:
            包含新颖性评估结果的字典
        """
        logger.info(f"正在评估论文新颖性: {paper['title']}")
        
        # 构建提示词
        prompt = self._build_novelty_prompt(paper, summary)
        
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
            
            # 解析评估结果
            assessment = self._parse_novelty_result(result)
            logger.info(f"新颖性评分: {assessment['score']}/10")
            return assessment
            
        except Exception as e:
            logger.error(f"评估论文新颖性时出错: {str(e)}")
            return {
                "score": 5.0,
                "level": "中等",
                "description": f"评估失败: {str(e)}",
                "strengths": [],
                "limitations": []
            }
    
    def _build_novelty_prompt(self, paper: Dict[str, Any], summary: str) -> str:
        """构建新颖性评估提示词
        
        Args:
            paper: 包含论文信息的字典
            summary: 论文的主要贡献摘要
            
        Returns:
            格式化的提示词
        """
        title = paper["title"]
        abstract = paper["summary"]
        
        # 提取论文全文中的关键部分（如果有）
        text_content = paper.get("text_content", "")
        intro_section = self._extract_introduction_section(text_content)
        related_work_section = self._extract_related_work_section(text_content)
        
        prompt = f"""
        请评估以下AI论文的新颖性和增量贡献。
        
        论文信息:
        标题: {title}
        摘要: {abstract}
        
        论文主要贡献:
        {summary}
        
        引言部分:
        {intro_section}
        
        相关工作部分:
        {related_work_section}
        
        请评估该论文相对于现有工作的新颖性和增量贡献。特别关注:
        1. 论文是提出全新方法，还是对现有方法的改进?
        2. 改进幅度有多大? 是革命性突破还是渐进式改进?
        3. 论文的创新点是在算法、模型、应用场景还是理论基础?
        4. 论文是否解决了领域内的重要挑战或开辟了新研究方向?
        
        请按照以下JSON格式提供评估结果:
        ```json
        {{
            "score": 7.5, // 新颖性评分，1-10分
            "level": "显著", // 新颖性水平: 低、中等、显著、突破性
            "description": "详细评估...",
            "strengths": ["创新点1", "创新点2"...],
            "limitations": ["局限性1", "局限性2"...]
        }}
        ```
        
        只需返回JSON格式的结果，不需要其他说明。
        """
        return prompt
    
    def _parse_novelty_result(self, result: str) -> Dict[str, Any]:
        """解析新颖性评估结果
        
        Args:
            result: LLM返回的评估结果文本
            
        Returns:
            解析后的评估结果字典
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
            assessment = json.loads(json_str)
            
            # 确保有所有必要的字段
            if "score" not in assessment:
                assessment["score"] = 5.0
            if "level" not in assessment:
                assessment["level"] = "中等"
            if "description" not in assessment:
                assessment["description"] = "无详细评估"
            if "strengths" not in assessment:
                assessment["strengths"] = []
            if "limitations" not in assessment:
                assessment["limitations"] = []
                
            return assessment
            
        except Exception as e:
            logger.error(f"解析新颖性评估结果时出错: {str(e)}")
            
            # 尝试提取评分
            score_match = re.search(r'(\d+(\.\d+)?)/10', result)
            score = float(score_match.group(1)) if score_match else 5.0
            
            # 尝试判断新颖性水平
            if "突破" in result or "革命" in result:
                level = "突破性"
            elif "显著" in result or "高" in result:
                level = "显著"
            elif "中等" in result:
                level = "中等"
            else:
                level = "低"
            
            # 失败情况下返回基本结果
            return {
                "score": score,
                "level": level,
                "description": result[:500],
                "strengths": [],
                "limitations": []
            }
    
    def _extract_introduction_section(self, text_content: str) -> str:
        """从论文全文中提取引言部分
        
        Args:
            text_content: 论文全文内容
            
        Returns:
            提取的引言部分文本
        """
        if not text_content:
            return ""
        
        # 使用正则表达式提取引言部分
        # 这里的模式可能需要根据实际论文格式进行调整
        intro_patterns = [
            r'(?:1\.|I\.|1\s)?\s*(?:Introduction|INTRODUCTION).*?(?=\n\s*(?:\d+\.|[A-Z]+\.|II\.))',  # 查找到下一个章节为止
            r'(?:Abstract|ABSTRACT).*?(?=\n\s*(?:\d+\.|[A-Z]+\.|II\.))',  # 如果没有明确的引言部分，使用摘要后到下一节之间的内容
        ]
        
        for pattern in intro_patterns:
            match = re.search(pattern, text_content, re.DOTALL)
            if match:
                intro = match.group(0)
                # 限制长度，只保留前1500个字符
                return intro[:1500] + ("..." if len(intro) > 1500 else "")
        
        # 如果没有找到明确的引言部分，返回前1500个字符
        return text_content[:1500] + "..."
    
    def _extract_related_work_section(self, text_content: str) -> str:
        """从论文全文中提取相关工作部分
        
        Args:
            text_content: 论文全文内容
            
        Returns:
            提取的相关工作部分文本
        """
        if not text_content:
            return ""
        
        # 使用正则表达式提取相关工作部分
        # 这里的模式可能需要根据实际论文格式进行调整
        related_work_patterns = [
            r'(?:\d+\.|[A-Z]+\.|II\.)?\s*(?:Related Work|RELATED WORK|Background|BACKGROUND|Previous Work|PREVIOUS WORK).*?(?=\n\s*(?:\d+\.|[A-Z]+\.|III\.))',
        ]
        
        for pattern in related_work_patterns:
            match = re.search(pattern, text_content, re.DOTALL)
            if match:
                related_work = match.group(0)
                # 限制长度，只保留前1500个字符
                return related_work[:1500] + ("..." if len(related_work) > 1500 else "")
        
        # 如果没有找到明确的相关工作部分，尝试在引言后找一段文本
        intro_end_match = re.search(r'(?:Introduction|INTRODUCTION).*?(?=\n\s*(?:\d+\.|[A-Z]+\.|II\.))', text_content, re.DOTALL)
        if intro_end_match:
            end_pos = intro_end_match.end()
            next_section = text_content[end_pos:end_pos+2000]
            return next_section[:1500] + "..."
        
        # 如果还是没找到，返回空字符串
        return ""
    
    def demo_run(self, title: str, abstract: str, summary: str, full_text: str = "") -> Dict[str, Any]:
        """演示运行，对示例论文进行新颖性评估"""
        mock_paper = {
            "title": title,
            "summary": abstract,
            "text_content": full_text
        }
        return self.assess_novelty(mock_paper, summary)
