# demo.py
import os
import logging
import json
from dotenv import load_dotenv
from agents.scraper import ArxivScraperAgent
from agents.summarizer import SummarizerAgent
from agents.classifier import ClassifierAgent
from agents.novelty_assessor import NoveltyAssessorAgent
from agents.scorer import ScorerAgent
from database.db_manager import DatabaseManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

def setup_config():
    """设置配置"""
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "interested_fields": [
            "大型语言模型优化与效率", 
            "多模态AI系统", 
            "AI安全与对齐", 
            "强化学习新方法",
            "生成式AI应用"
        ],
        "arxiv_categories": ["cs.AI"], # "arxiv_categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO"],
        "max_papers_per_run": 5,  # 演示时限制数量
        "database_path": "demo_papers.db",
        "llm_config": {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,
        }
    }
    return config

def demo_scraper(config):
    """演示数据抓取Agent"""
    logger.info("=== 演示数据抓取Agent ===")
    scraper = ArxivScraperAgent(config)
    
    # 修改为获取最近7天的论文，增加成功率
    papers = scraper.get_papers(days_range=7)
    
    if papers:
        logger.info(f"获取到{len(papers)}篇论文")
        sample_paper = papers[0]
        logger.info(f"示例论文标题: {sample_paper['title']}")
        logger.info(f"示例论文作者: {sample_paper['authors']}")
        logger.info(f"示例论文分类: {sample_paper['categories']}")
        logger.info(f"示例论文摘要: {sample_paper['summary'][:200]}...")
    else:
        logger.info("未获取到论文")
    return papers

def demo_summarizer(config, paper):
    """演示摘要Agent"""
    logger.info("\n=== 演示摘要Agent ===")
    summarizer = SummarizerAgent(config)
    summary = summarizer.summarize_paper(paper)
    logger.info(f"生成的摘要:\n{summary}")
    return summary

def demo_classifier(config, paper, summary):
    """演示分类Agent"""
    logger.info("\n=== 演示分类Agent ===")
    classifier = ClassifierAgent(config)
    classification = classifier.classify_paper(paper, summary)
    logger.info(f"分类结果: {classification['category']}")
    logger.info(f"分类置信度: {classification['confidence']}")
    logger.info(f"分类理由: {classification['rationale']}")
    return classification

def demo_novelty_assessor(config, paper, summary):
    """演示新颖性评估Agent"""
    logger.info("\n=== 演示新颖性评估Agent ===")
    assessor = NoveltyAssessorAgent(config)
    assessment = assessor.assess_novelty(paper, summary)
    logger.info(f"新颖性评分: {assessment['score']}/10")
    logger.info(f"新颖性水平: {assessment['level']}")
    logger.info(f"新颖性评估: {assessment['description'][:200]}...")
    logger.info(f"优势: {assessment['strengths']}")
    logger.info(f"局限性: {assessment['limitations']}")
    return assessment

def demo_scorer(config, paper, summary, classification, assessment):
    """演示评分Agent"""
    logger.info("\n=== 演示评分Agent ===")
    scorer = ScorerAgent(config)
    score, rationale = scorer.score_paper(paper, summary, classification, assessment)
    logger.info(f"论文评分: {score}/10")
    logger.info(f"评分理由: {rationale[:200]}...")
    return score, rationale

def demo_database(config):
    """演示数据库操作"""
    logger.info("\n=== 演示数据库操作 ===")
    db_manager = DatabaseManager(config["database_path"])
    db_manager.initialize_database()
    
    # 创建示例论文结果
    paper_result = {
        "paper_id": "demo123",
        "title": "示例论文: 大型语言模型的新型优化方法",
        "authors": "张三, 李四",
        "published_date": "2023-01-15",
        "processed_date": "2023-01-16",
        "summary": "这篇论文提出了一种新的方法来优化大型语言模型的训练过程...",
        "classification": {
            "category": "大型语言模型优化与效率",
            "confidence": 0.95,
            "rationale": "论文直接关注LLM优化技术"
        },
        "novelty_assessment": {
            "score": 8.5,
            "level": "显著",
            "description": "该论文提出的方法在现有工作基础上有明显改进",
            "strengths": ["创新算法", "显著性能提升"],
            "limitations": ["计算成本高", "仅在特定任务上测试"]
        },
        "score": 8.0,
        "scoring_rationale": "该论文在LLM优化领域提出了创新方法，实验结果显示性能提升明显..."
    }
    
    # 保存到数据库
    success = db_manager.save_paper_analysis(paper_result)
    logger.info(f"保存结果: {'成功' if success else '失败'}")
    
    # 检索论文
    retrieved_paper = db_manager.get_paper_by_id("demo123")
    if retrieved_paper:
        logger.info(f"成功检索论文: {retrieved_paper['title']}")
    else:
        logger.info("未找到论文")
    
    # 获取统计信息
    stats = db_manager.get_statistics()
    logger.info(f"数据库统计: 共{stats['total_papers']}篇论文，平均分数: {stats['avg_score']}")

def run_full_demo():
    """运行完整演示"""
    config = setup_config()
    
    # 确保设置了API密钥
    if not config["openai_api_key"]:
        logger.error("请设置OPENAI_API_KEY环境变量")
        return
    
    # 1. 演示数据抓取
    papers = demo_scraper(config)
    if not papers:
        logger.error("未获取到论文，演示中止")
        return
    
    # 选择第一篇论文进行演示
    sample_paper = papers[0]
    
    # 2. 演示摘要生成
    summary = demo_summarizer(config, sample_paper)
    
    # 3. 演示分类
    classification = demo_classifier(config, sample_paper, summary)
    
    # 4. 演示新颖性评估
    assessment = demo_novelty_assessor(config, sample_paper, summary)
    
    # 5. 演示评分
    score, rationale = demo_scorer(config, sample_paper, summary, classification, assessment)
    
    # 6. 演示数据库操作
    demo_database(config)
    
    # 7. 保存当前论文的分析结果
    db_manager = DatabaseManager(config["database_path"])
    paper_result = {
        "paper_id": sample_paper["id"],
        "title": sample_paper["title"],
        "authors": sample_paper["authors"],
        "published_date": sample_paper["published"],
        "processed_date": "2023-01-16",  # 示例日期
        "summary": summary,
        "classification": classification,
        "novelty_assessment": assessment,
        "score": score,
        "scoring_rationale": rationale
    }
    db_manager.save_paper_analysis(paper_result)
    
    logger.info("\n=== 演示完成 ===")
    logger.info(f"论文《{sample_paper['title']}》分析结果:")
    logger.info(f"分类: {classification['category']}")
    logger.info(f"新颖性: {assessment['score']}/10 ({assessment['level']})")
    logger.info(f"最终评分: {score}/10")

if __name__ == "__main__":
    run_full_demo()
