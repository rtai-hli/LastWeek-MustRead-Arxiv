import os
import json
import datetime
import logging
import schedule
import time
import pandas as pd
import autogen
from dotenv import load_dotenv
from agents.coordinator import CoordinatorAgent
from agents.scraper import ArxivScraperAgent
from agents.summarizer import SummarizerAgent
from agents.classifier import ClassifierAgent
from agents.novelty_assessor import NoveltyAssessorAgent
from agents.scorer import ScorerAgent
from database.db_manager import DatabaseManager

# 加载环境变量
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        "max_papers_per_run": 10,  # 开发测试时限制数量
        "database_path": "papers.db",
        "llm_config": {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,
        }
    }
    return config

def run_analysis_pipeline(config):
    """运行完整的分析流程"""
    logger.info("启动论文分析流程")
    
    # 初始化数据库
    db_manager = DatabaseManager(config["database_path"])
    db_manager.initialize_database()
    
    # 创建智能体
    coordinator = CoordinatorAgent(config)
    scraper = ArxivScraperAgent(config)
    summarizer = SummarizerAgent(config)
    classifier = ClassifierAgent(config)
    novelty_assessor = NoveltyAssessorAgent(config)
    scorer = ScorerAgent(config)
    
    # 获取最近的论文
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    logger.info(f"正在获取{today}及近期的论文")
    papers = scraper.get_papers(days_range=7)  # 获取最近7天的论文
    
    if not papers:
        logger.info("近期没有新论文，流程结束")
        return
    
    logger.info(f"今日获取了{len(papers)}篇论文")
    
    # 处理每篇论文
    results = []
    for i, paper in enumerate(papers[:config["max_papers_per_run"]]):
        logger.info(f"处理论文 {i+1}/{min(len(papers), config['max_papers_per_run'])}: {paper['title']}")
        
        # 摘要分析
        summary = summarizer.summarize_paper(paper)
        
        # 分类
        classification = classifier.classify_paper(paper, summary)
        
        # 新颖性评估
        novelty_assessment = novelty_assessor.assess_novelty(paper, summary)
        
        # 评分
        score, scoring_rationale = scorer.score_paper(paper, summary, classification, novelty_assessment)
        
        # 保存结果
        paper_result = {
            "paper_id": paper["id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "published_date": paper["published"],
            "summary": summary,
            "classification": classification,
            "novelty_assessment": novelty_assessment,
            "score": score,
            "scoring_rationale": scoring_rationale,
            "processed_date": today
        }
        results.append(paper_result)
        
        # 存入数据库
        db_manager.save_paper_analysis(paper_result)
    
    # 生成每日报告
    daily_report = pd.DataFrame(results).sort_values(by="score", ascending=False)
    report_path = f"reports/daily_report_{today}.csv"
    os.makedirs("reports", exist_ok=True)
    daily_report.to_csv(report_path, index=False, encoding="utf-8")
    
    logger.info(f"分析完成，报告已保存至 {report_path}")
    return daily_report

def schedule_daily_run():
    """设置每日定时运行"""
    config = setup_config()
    # 每天凌晨2点运行 (arXiv通常在北美时间下午更新)
    schedule.every().day.at("02:00").do(run_analysis_pipeline, config)
    
    logger.info("系统已启动，将在每天02:00运行")
    while True:
        schedule.run_pending()
        time.sleep(60)

def run_once():
    """立即运行一次完整分析"""
    config = setup_config()
    return run_analysis_pipeline(config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="arXiv论文分析系统")
    parser.add_argument("--mode", choices=["schedule", "once"], default="once",
                        help="运行模式: schedule (定时运行) 或 once (立即运行一次)")
    args = parser.parse_args()
    
    if args.mode == "schedule":
        schedule_daily_run()
    else:
        run_once()
