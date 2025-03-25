# agents/classifier.py
import logging
import autogen
from typing import Dict, Any, List
import openai

logger = logging.getLogger(__name__)

class ClassifierAgent:
    """负责将论文分类到用户定义的研究领域的Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config["llm_config"]
        self.interested_fields = config["interested_fields"]
        
        # 创建AutoGen智能体配置
        self.agent_config = {
            "name": "ClassifierAgent",
            "llm_config": self.llm_config,
            "system_message": f"""
            你是一位专业的AI论文分类专家，擅长将论文归类到特定的研究领域。
            你需要将论文分类到以下五个领域之一：
            {', '.join(self.interested_fields)}
            
            如果论文跨越多个领域，请选择最主要的一个。
            如果论文不属于以上任何一个领域，请标记为"其他"。
            
            为每篇论文提供分类结果时，请解释你的分类理由。
            """
        }
        
        # 创建AutoGen智能体实例
        self.agent = autogen.AssistantAgent(**self.agent_config)
        self.user_proxy = autogen.UserProxyAgent(
            name="ClassifierProxy",
            human_input_mode="NEVER",
            system_message="你代表分类智能体与其他智能体通信。"
        )
        
        # 初始化OpenAI客户端（直接API调用备选方案）
        self.client = openai.OpenAI(api_key=config["openai_api_key"])
    
    def classify_paper(self, paper: Dict[str, Any], summary: str) -> Dict[str, Any]:
        """将论文分类到感兴趣的研究领域
        
        Args:
            paper: 包含论文信息的字典
            summary: 论文的主要贡献摘要（由摘要Agent生成）
            
        Returns:
            包含分类结果和理由的字典
        """
        logger.info(f"正在对论文进行分类: {paper['title']}")
        
        # 构建提示词
        prompt = self._build_classification_prompt(paper, summary)
        
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
            
            # 解析分类结果
            classification = self._parse_classification_result(result)
            logger.info(f"分类结果: {classification['category']}")
            return classification
            
        except Exception as e:
            logger.error(f"论文分类时出错: {str(e)}")
            return {
                "category": "未分类",
                "confidence": 0.0,
                "rationale": f"分类失败: {str(e)}"
            }
    
    def _build_classification_prompt(self, paper: Dict[str, Any], summary: str) -> str:
        """构建分类提示词
        
        Args:
            paper: 包含论文信息的字典
            summary: 论文的主要贡献摘要
            
        Returns:
            格式化的提示词
        """
        title = paper["title"]
        abstract = paper["summary"]
        
        prompt = f"""
        请将以下AI论文分类到我们感兴趣的五个研究领域之一。
        
        可选的研究领域:
        {', '.join([f"{i+1}. {field}" for i, field in enumerate(self.interested_fields)])}
        
        如果论文不属于以上任何领域，请分类为"其他"。
        
        论文信息:
        标题: {title}
        摘要: {abstract}
        
        论文主要贡献:
        {summary}
        
        请按照以下JSON格式提供分类结果:
        ```json
        {{
            "category": "选择的领域名称",
            "confidence": 0.85, // 分类的置信度，0-1之间的小数
            "rationale": "分类理由的详细解释..."
        }}
        ```
        
        只需返回JSON格式的结果，不需要其他说明。
        """
        return prompt
    
    def _parse_classification_result(self, result: str) -> Dict[str, Any]:
        """解析分类结果
        
        Args:
            result: LLM返回的分类结果文本
            
        Returns:
            解析后的分类结果字典
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
            classification = json.loads(json_str)
            
            # 确保有所有必要的字段
            if "category" not in classification:
                classification["category"] = "未分类"
            if "confidence" not in classification:
                classification["confidence"] = 0.0
            if "rationale" not in classification:
                classification["rationale"] = "无分类理由"
                
            return classification
            
        except Exception as e:
            logger.error(f"解析分类结果时出错: {str(e)}")
            # 尝试简单提取类别名
            for field in self.interested_fields:
                if field in result:
                    return {
                        "category": field,
                        "confidence": 0.5,
                        "rationale": f"解析错误，但发现匹配类别: {field}"
                    }
            
            # 失败情况下返回默认值
            return {
                "category": "其他",
                "confidence": 0.0,
                "rationale": f"无法解析分类结果: {result[:100]}..."
            }
    
    def demo_run(self, title: str, abstract: str, summary: str) -> Dict[str, Any]:
        """演示运行，对示例论文进行分类"""
        mock_paper = {
            "title": title,
            "summary": abstract
        }
        return self.classify_paper(mock_paper, summary)
