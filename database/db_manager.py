# database/db_manager.py
import logging
import sqlite3
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """管理论文分析结果的数据库"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def initialize_database(self):
        """初始化数据库结构"""
        try:
            # 确保数据库目录存在
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            
            # 创建数据库连接
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建表结构
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                published_date TEXT,
                processed_date TEXT,
                summary TEXT,
                classification TEXT,
                novelty_assessment TEXT,
                score REAL,
                scoring_rationale TEXT
            )
            ''')
            
            # 提交更改
            conn.commit()
            logger.info(f"数据库初始化成功: {self.db_path}")
            
        except Exception as e:
            logger.error(f"初始化数据库时出错: {str(e)}")
            
        finally:
            # 关闭连接
            if 'conn' in locals():
                conn.close()
    
    def save_paper_analysis(self, paper_result: Dict[str, Any]) -> bool:
        """保存论文分析结果到数据库
        
        Args:
            paper_result: 包含论文分析结果的字典
            
        Returns:
            保存是否成功
        """
        try:
            # 创建数据库连接
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 准备数据
            paper_id = paper_result["paper_id"]
            title = paper_result["title"]
            authors = paper_result["authors"]
            published_date = paper_result["published_date"]
            processed_date = paper_result["processed_date"]
            summary = paper_result["summary"]
            classification = json.dumps(paper_result["classification"], ensure_ascii=False)
            novelty_assessment = json.dumps(paper_result["novelty_assessment"], ensure_ascii=False)
            score = paper_result["score"]
            scoring_rationale = paper_result["scoring_rationale"]
            
            # 插入或更新数据
            cursor.execute('''
            INSERT OR REPLACE INTO papers (
                id, title, authors, published_date, processed_date, 
                summary, classification, novelty_assessment, score, scoring_rationale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper_id, title, authors, published_date, processed_date, 
                summary, classification, novelty_assessment, score, scoring_rationale
            ))
            
            # 提交更改
            conn.commit()
            logger.info(f"成功保存论文分析结果: {title}")
            return True
            
        except Exception as e:
            logger.error(f"保存论文分析结果时出错: {str(e)}")
            return False
            
        finally:
            # 关闭连接
            if 'conn' in locals():
                conn.close()
    
    def get_papers_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """获取特定日期处理的论文
        
        Args:
            date_str: 日期字符串，格式为YYYY-MM-DD
            
        Returns:
            包含论文信息的字典列表
        """
        try:
            # 创建数据库连接
            conn = sqlite3.connect(self.db_path)
            
            # 查询数据
            query = f"SELECT * FROM papers WHERE processed_date = ?"
            papers_df = pd.read_sql_query(query, conn, params=(date_str,))
            
            # 处理JSON字段
            papers_df["classification"] = papers_df["classification"].apply(
                lambda x: json.loads(x) if x else {})
            papers_df["novelty_assessment"] = papers_df["novelty_assessment"].apply(
                lambda x: json.loads(x) if x else {})
            
            # 转换为字典列表
            papers = papers_df.to_dict(orient="records")
            logger.info(f"成功获取{date_str}的{len(papers)}篇论文")
            return papers
            
        except Exception as e:
            logger.error(f"获取论文时出错: {str(e)}")
            return []
            
        finally:
            # 关闭连接
            if 'conn' in locals():
                conn.close()
    
    def get_top_papers(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取评分最高的n篇论文
        
        Args:
            n: 要返回的论文数量
            
        Returns:
            包含论文信息的字典列表
        """
        try:
            # 创建数据库连接
            conn = sqlite3.connect(self.db_path)
            
            # 查询数据
            query = f"SELECT * FROM papers ORDER BY score DESC LIMIT {n}"
            papers_df = pd.read_sql_query(query, conn)
            
            # 处理JSON字段
            papers_df["classification"] = papers_df["classification"].apply(
                lambda x: json.loads(x) if x else {})
            papers_df["novelty_assessment"] = papers_df["novelty_assessment"].apply(
                lambda x: json.loads(x) if x else {})
            
            # 转换为字典列表
            papers = papers_df.to_dict(orient="records")
            logger.info(f"成功获取评分最高的{len(papers)}篇论文")
            return papers
            
        except Exception as e:
            logger.error(f"获取论文时出错: {str(e)}")
            return []
            
        finally:
            # 关闭连接
            if 'conn' in locals():
                conn.close()
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取特定论文
        
        Args:
            paper_id: 论文ID
            
        Returns:
            包含论文信息的字典，如果未找到则返回None
        """
        try:
            # 创建数据库连接
            conn = sqlite3.connect(self.db_path)
            
            # 查询数据
            query = "SELECT * FROM papers WHERE id = ?"
            papers_df = pd.read_sql_query(query, conn, params=(paper_id,))
            
            if len(papers_df) == 0:
                logger.warning(f"未找到ID为{paper_id}的论文")
                return None
            
            # 处理JSON字段
            papers_df["classification"] = papers_df["classification"].apply(
                lambda x: json.loads(x) if x else {})
            papers_df["novelty_assessment"] = papers_df["novelty_assessment"].apply(
                lambda x: json.loads(x) if x else {})
            
            # 转换为字典
            paper = papers_df.iloc[0].to_dict()
            logger.info(f"成功获取论文: {paper['title']}")
            return paper
            
        except Exception as e:
            logger.error(f"获取论文时出错: {str(e)}")
            return None
            
        finally:
            # 关闭连接
            if 'conn' in locals():
                conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息
        
        Returns:
            包含统计信息的字典
        """
        try:
            # 创建数据库连接
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取总论文数
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]
            
            # 获取平均分数
            cursor.execute("SELECT AVG(score) FROM papers")
            avg_score = cursor.fetchone()[0]
            
            # 获取最高分数
            cursor.execute("SELECT MAX(score) FROM papers")
            max_score = cursor.fetchone()[0]
            
            # 获取最低分数
            cursor.execute("SELECT MIN(score) FROM papers")
            min_score = cursor.fetchone()[0]
            
            # 获取分类统计
            cursor.execute("SELECT classification, COUNT(*) FROM papers GROUP BY classification")
            classification_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 获取日期统计
            cursor.execute("SELECT processed_date, COUNT(*) FROM papers GROUP BY processed_date")
            date_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            stats = {
                "total_papers": total_papers,
                "avg_score": avg_score,
                "max_score": max_score,
                "min_score": min_score,
                "classification_stats": classification_stats,
                "date_stats": date_stats
            }
            
            logger.info(f"成功获取数据库统计信息: 共{total_papers}篇论文")
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息时出错: {str(e)}")
            return {
                "total_papers": 0,
                "avg_score": 0,
                "max_score": 0,
                "min_score": 0,
                "classification_stats": {},
                "date_stats": {}
            }
            
        finally:
            # 关闭连接
            if 'conn' in locals():
                conn.close()
