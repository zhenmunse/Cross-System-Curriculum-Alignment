# -*- coding: utf-8 -*-
"""
文本向量化模块。
默认使用 TF-IDF 实现，并提供可插拔的接口，便于未来扩展到其他向量化方法（如 SBERT）。
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class TextEmbedder:
    """
    文本向量化器基类，定义通用接口。
    """
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit_transform(self, texts):
        """
        训练并转换文本为向量。
        
        Args:
            texts (list of str): 待处理的文本列表。
        
        Returns:
            scipy.sparse.csr_matrix or numpy.ndarray: 文本向量矩阵。
        """
        raise NotImplementedError

    def calculate_similarity(self, vectors_source, vectors_target):
        """
        计算两组向量之间的余弦相似度。
        
        Args:
            vectors_source: 源向量矩阵。
            vectors_target: 目标向量矩阵。
            
        Returns:
            numpy.ndarray: 相似度矩阵。
        """
        return cosine_similarity(vectors_source, vectors_target)


class TfidfEmbedder(TextEmbedder):
    """
    使用 TF-IDF 实现的文本向量化器。
    """
    def __init__(self, random_state=42):
        super().__init__(random_state)
        # 使用英文停用词
        # 注意：TfidfVectorizer 不接受 random_state 参数，因为它是确定性的
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fit_transform(self, texts):
        """
        使用 TF-IDF 训练并转换文本。
        """
        return self.vectorizer.fit_transform(texts)

def get_embedder(method='tfidf', **kwargs) -> TextEmbedder:
    """
    工厂函数，用于获取指定类型的向量化器实例。
    
    Args:
        method (str): 向量化方法 ('tfidf', etc.)。
        **kwargs: 传递给向量化器构造函数的参数。
        
    Returns:
        TextEmbedder: 向量化器实例。
    """
    if method == 'tfidf':
        return TfidfEmbedder(**kwargs)
    # 未来可在此处添加其他向量化方法，例如：
    # elif method == 'sbert':
    #     return SbertEmbedder(**kwargs)
    else:
        raise ValueError(f"不支持的向量化方法: {method}")
