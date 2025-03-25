# agents/summarizer.py
import logging
import autogen
from typing import Dict, Any, List
import openai

logger = logging.getLogger(__name__)

class SummarizerAgent:
    """负责提取论文主要贡献和创新点的摘要Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config["llm_config"]
        
        # 创建AutoGen智能体配置
        self.agent_config = {
            "name": "SummarizerAgent",
            "llm_config": self.llm_config,
            "system_message": """
            你是一位专业的AI论文摘要专家，擅长从论文中提取关键的研究贡献和创新点。
            你的任务是阅读论文内容，并提取出论文的主要贡献、创新方法和关键发现。
            你需要关注论文中的"我们提出了..."、"贡献包括..."、"本文的创新点..."等关键描述。
            请保持摘要简洁、准确、全面，突出论文的核心创新和学术价值。
            """
        }
        
        # 创建AutoGen智能体实例
        self.agent = autogen.AssistantAgent(**self.agent_config)
        self.user_proxy = autogen.UserProxyAgent(
            name="SummarizerProxy",
            human_input_mode="NEVER",
            system_message="你代表摘要智能体与其他智能体通信。"
        )
        
        # 初始化OpenAI客户端（直接API调用备选方案）
        self.client = openai.OpenAI(api_key=config["openai_api_key"])
    
    def summarize_paper(self, paper: Dict[str, Any]) -> str:
        """生成论文的主要贡献摘要
        
        Args:
            paper: 包含论文信息的字典，包括标题、作者、摘要和正文
            
        Returns:
            论文主要贡献的摘要
        """
        logger.info(f"正在生成论文摘要: {paper['title']}")
        
        # 构建提示词
        prompt = self._build_summarization_prompt(paper)
        
        try:
            # 方法1: 使用AutoGen框架（适合复杂互动）
            if False:  # 这里设为False，使用方法2，因为这个任务较简单
                self.user_proxy.initiate_chat(self.agent, message=prompt)
                # 从chat_history中提取摘要
                summary = self.user_proxy.chat_history[-1][-1]["content"]
            # 方法2: 直接调用OpenAI API（更简单，速度更快）
            else:
                response = self.client.chat.completions.create(
                    model=self.llm_config["model"],
                    temperature=self.llm_config["temperature"],
                    messages=[
                        {"role": "system", "content": self.agent_config["system_message"]},
                        {"role": "user", "content": prompt}
                    ]
                )
                summary = response.choices[0].message.content
                
            logger.info(f"成功生成摘要: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.error(f"生成摘要时出错: {str(e)}")
            return f"摘要生成失败: {str(e)}"
    
    def _build_summarization_prompt(self, paper: Dict[str, Any]) -> str:
        """构建摘要提示词
        
        Args:
            paper: 包含论文信息的字典
            
        Returns:
            格式化的提示词
        """
        # 提取论文的关键部分用于摘要
        title = paper["title"]
        authors = paper["authors"]
        abstract = paper["summary"]
        
        # 如果有全文，取前5000个字符和后2000个字符（通常包含介绍和结论）
        text_content = paper.get("text_content", "")
        if len(text_content) > 7000:
            text_sample = text_content[:5000] + "\n...\n" + text_content[-2000:]
        else:
            text_sample = text_content
        
        prompt = f"""
        请分析以下AI论文，并提取出论文的主要贡献、创新方法和关键发现。
        
        标题: {title}
        作者: {authors}
        摘要: {abstract}
        
        论文内容:
        {text_sample}
        
        请按照以下结构提供一个简洁但全面的中文摘要:
        
        1. 研究问题: [简要描述论文解决的主要问题]
        2. 主要方法/技术: [概述论文提出的方法或技术]
        3. 核心创新点: [列出论文的主要创新点和贡献]
        4. 主要发现/结果: [总结论文的关键发现和实验结果]
        5. 潜在影响: [分析这项工作对AI领域可能的影响]
        
        摘要应该强调论文的独特贡献，而不是一般性描述。请特别注意寻找论文中表示创新的关键句子，例如"本文的主要贡献..."、"我们提出了..."、"与现有方法相比..."等。
        """
        return prompt
    
    def demo_run(self, text: str) -> str:
        """演示运行，使用示例文本生成摘要"""
        mock_paper = {
            "title": "示例AI论文标题",
            "authors": "张三, 李四, 王五",
            "summary": "这是一篇关于大型语言模型优化的论文摘要...",
            "text_content": text
        }
        return self.summarize_paper(mock_paper)
