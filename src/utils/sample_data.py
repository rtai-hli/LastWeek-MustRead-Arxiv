"""Utility module for generating sample paper data."""

import datetime
from typing import List, Dict, Any

def get_sample_papers() -> List[Dict[str, Any]]:
    """Generate sample papers for testing and demonstration.
    
    Returns:
        List of sample paper dictionaries
    """
    return [
        {
            "id": "sample1",
            "title": "Large Language Models as Zero-Shot Reasoners for Biomedical Problems",
            "authors": ["Jane Doe", "John Smith", "Robert Johnson"],
            "abstract": "This paper explores the application of large language models (LLMs) to challenging biomedical reasoning tasks without task-specific fine-tuning. We show that with appropriate prompting techniques, LLMs can achieve competitive performance on biomedical question answering, medical diagnosis, and drug discovery tasks. Our approach introduces a novel chain-of-thought prompting strategy specifically designed for biomedical contexts, allowing models to reason step-by-step through complex medical scenarios. Experiments across multiple biomedical benchmarks demonstrate that our method outperforms traditional fine-tuning approaches while requiring significantly less task-specific data.",
            "published": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            "updated": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            "primary_category": "cs.AI",
            "categories": ["cs.AI", "cs.CL", "q-bio.QM"],
            "pdf_url": "https://arxiv.org/pdf/sample1.pdf",
            "links": ["https://arxiv.org/abs/sample1"],
            "comment": "20 pages, 8 figures"
        },
        {
            "id": "sample2",
            "title": "Vision-Language Transformer with Adaptive Multi-Granularity Attention",
            "authors": ["Wei Zhang", "Li Chen", "Mei Wang"],
            "abstract": "We present a novel vision-language transformer architecture that processes visual and textual information at multiple levels of granularity simultaneously. Our model, AMG-Transformer, dynamically adjusts attention across different semantic levels, from fine-grained pixel-word interactions to coarse document-image relationships. Experiments on visual question answering, image captioning, and visual reasoning tasks show significant improvements over previous vision-language models. The adaptive multi-granularity mechanism proves particularly effective for tasks requiring both detailed visual understanding and high-level semantic reasoning.",
            "published": (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
            "updated": (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
            "primary_category": "cs.CV",
            "categories": ["cs.CV", "cs.AI", "cs.CL"],
            "pdf_url": "https://arxiv.org/pdf/sample2.pdf",
            "links": ["https://arxiv.org/abs/sample2"],
            "comment": "15 pages, 6 figures, 4 tables"
        },
        {
            "id": "sample3",
            "title": "Efficient Reinforcement Learning with Adaptive State Abstraction",
            "authors": ["Hiroshi Tanaka", "Maria Garcia", "David Wilson"],
            "abstract": "This paper introduces a novel reinforcement learning algorithm that dynamically adapts its state representation based on the learning progress. Our approach, Adaptive State Abstraction (ASA), automatically identifies which state features are relevant for different parts of the environment, creating simplified representations that accelerate learning while maintaining performance. We demonstrate ASA's effectiveness on complex control tasks and video game environments, showing significant improvements in sample efficiency compared to state-of-the-art methods. Our analysis reveals that ASA discovers meaningful state abstractions that align with human intuition about relevant features.",
            "published": (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
            "updated": (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
            "primary_category": "cs.LG",
            "categories": ["cs.LG", "cs.AI", "cs.RO"],
            "pdf_url": "https://arxiv.org/pdf/sample3.pdf",
            "links": ["https://arxiv.org/abs/sample3"],
            "comment": "18 pages, 10 figures, ICML 2024"
        }
    ] 