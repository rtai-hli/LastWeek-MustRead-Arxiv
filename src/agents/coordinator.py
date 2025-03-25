# agents/coordinator.py
import logging
import autogen
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CoordinatorAgent:
    """中央协调Agent，负责管理其他Agent的工作流程"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config["llm_config"]
        
        # 创建AutoGen智能体配置
        self.agent_config = {
            "name": "CoordinatorAgent",
            "llm_config": self.llm_config,
            "system_message": """
            你是一个AI论文分析系统的中央协调者。
            你负责协调多个专业智能体之间的工作流程，确保数据顺利流转，并解决可能出现的冲突。
            在与其他智能体通信时，保持简洁清晰，聚焦于当前任务和所需的信息。
            """
        }
        
        # 创建AutoGen智能体实例
        self.agent = autogen.AssistantAgent(**self.agent_config)
        self.user_proxy = autogen.UserProxyAgent(
            name="CoordinatorProxy",
            human_input_mode="NEVER",
            system_message="你代表协调智能体与其他智能体通信。"
        )
    
    def coordinate_workflow(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """协调完整的工作流程"""
        logger.info(f"协调工作流程开始，处理{len(papers)}篇论文")
        
        results = []
        for paper in papers:
            try:
                # 构建任务描述
                task_description = f"""
                需要处理的论文:
                标题: {paper['title']}
                作者: {paper['authors']}
                摘要: {paper['summary']}
                全文: [全文内容已加载]
                
                请协调以下工作流程:
                1. 让摘要Agent生成论文主要贡献的摘要
                2. 让分类Agent将论文分类到五个领域之一
                3. 让增量贡献评估Agent评估论文的新颖性
                4. 让评分Agent基于以上信息给论文评分
                
                请确保每个步骤都成功完成后再进行下一步，并收集所有结果。
                """
                
                self.user_proxy.initiate_chat(self.agent, message=task_description)
                
                # 实际工作流程将在各个具体Agent实现中完成
                # 这里只是协调框架的示例
                
                logger.info(f"完成论文处理: {paper['title']}")
                
            except Exception as e:
                logger.error(f"处理论文时出错: {str(e)}")
        
        return results
    
    def resolve_conflicts(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """解决智能体之间可能出现的冲突"""
        conflict_message = f"""
        以下是各个智能体的输出，其中可能存在冲突:
        
        摘要Agent: {agent_outputs.get('summary', 'N/A')}
        分类Agent: {agent_outputs.get('classification', 'N/A')}
        新颖性评估Agent: {agent_outputs.get('novelty', 'N/A')}
        评分Agent: {agent_outputs.get('score', 'N/A')}
        
        请分析这些输出中是否存在冲突，如果有，请给出解决方案。
        """
        
        self.user_proxy.initiate_chat(self.agent, message=conflict_message)
        # 在实际实现中，这里会解析智能体的回复并做出相应处理
        
        return agent_outputs  # 返回可能经过调整的输出
