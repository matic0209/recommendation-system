"""
基于内容的推荐算法
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class ContentBasedBase(ABC):
    """基于内容推荐基类"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.item_features = None
        self.item_similarity = None
        self.vectorizer = None
        self.scaler = None
    
    @abstractmethod
    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """训练模型"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """为用户推荐物品"""
        pass
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Dict[str, Any]]:
        """获取相似物品"""
        if item_id not in self.item_similarity.index:
            return []
        
        similar_items = self.item_similarity[item_id].sort_values(ascending=False)[1:n_similar+1]
        
        return [
            {
                'item_id': similar_item,
                'similarity': similarity,
                'algorithm': 'content_based'
            }
            for similar_item, similarity in similar_items.items()
        ]

class TFIDFContentBased(ContentBasedBase):
    """基于TF-IDF的文本内容推荐"""
    
    def __init__(self, config):
        super().__init__(config)
        self.max_features = 1000
        self.stop_words = 'english'
    
    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """训练TF-IDF内容推荐模型"""
        # 准备文本特征
        text_features = []
        for _, item in items_df.iterrows():
            text = f"{item.get('title', '')} {item.get('description', '')} {item.get('category', '')}"
            text_features.append(text)
        
        # 训练TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(text_features)
        
        # 计算物品相似度矩阵
        self.item_similarity = pd.DataFrame(
            cosine_similarity(tfidf_matrix),
            index=items_df['item_id'],
            columns=items_df['item_id']
        )
        
        self.logger.info("TF-IDF内容推荐模型训练完成")
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """基于用户历史评分推荐相似物品"""
        # 获取用户历史评分的物品
        user_items = self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'].tolist()
        
        if not user_items:
            return []
        
        # 计算用户对所有物品的相似度分数
        item_scores = {}
        for item_id in self.item_similarity.index:
            if item_id not in user_items:
                score = 0
                for user_item in user_items:
                    if user_item in self.item_similarity.columns:
                        score += self.item_similarity.loc[item_id, user_item]
                item_scores[item_id] = score / len(user_items)
        
        # 按分数排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'item_id': item_id,
                'score': score,
                'algorithm': 'tfidf_content_based'
            }
            for item_id, score in sorted_items[:n_recommendations]
        ]

class HybridContentBased(ContentBasedBase):
    """混合内容推荐（文本+数值特征）"""
    
    def __init__(self, config):
        super().__init__(config)
        self.text_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.scaler = StandardScaler()
    
    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """训练混合内容推荐模型"""
        self.ratings_df = ratings_df
        
        # 准备特征
        features = []
        
        # 文本特征
        text_features = []
        for _, item in items_df.iterrows():
            text = f"{item.get('title', '')} {item.get('description', '')}"
            text_features.append(text)
        
        tfidf_matrix = self.text_vectorizer.fit_transform(text_features)
        features.append(tfidf_matrix.toarray())
        
        # 数值特征
        numeric_features = ['price']
        for feature in numeric_features:
            if feature in items_df.columns:
                values = items_df[feature].fillna(items_df[feature].median()).values.reshape(-1, 1)
                scaled_values = self.scaler.fit_transform(values)
                features.append(scaled_values)
        
        # 分类特征（one-hot编码）
        categorical_features = ['category', 'brand']
        for feature in categorical_features:
            if feature in items_df.columns:
                one_hot = pd.get_dummies(items_df[feature], prefix=feature)
                features.append(one_hot.values)
        
        # 合并所有特征
        if features:
            combined_features = np.hstack(features)
        else:
            combined_features = tfidf_matrix.toarray()
        
        # 计算物品相似度矩阵
        self.item_similarity = pd.DataFrame(
            cosine_similarity(combined_features),
            index=items_df['item_id'],
            columns=items_df['item_id']
        )
        
        self.logger.info("混合内容推荐模型训练完成")
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """基于用户历史评分推荐相似物品"""
        # 获取用户历史评分的物品
        user_items = self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'].tolist()
        
        if not user_items:
            return []
        
        # 计算用户对所有物品的相似度分数
        item_scores = {}
        for item_id in self.item_similarity.index:
            if item_id not in user_items:
                score = 0
                for user_item in user_items:
                    if user_item in self.item_similarity.columns:
                        score += self.item_similarity.loc[item_id, user_item]
                item_scores[item_id] = score / len(user_items)
        
        # 按分数排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'item_id': item_id,
                'score': score,
                'algorithm': 'hybrid_content_based'
            }
            for item_id, score in sorted_items[:n_recommendations]
        ]

class CategoryBasedContent(ContentBasedBase):
    """基于类别的内容推荐"""
    
    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """训练基于类别的推荐模型"""
        self.ratings_df = ratings_df
        
        # 创建类别-物品映射
        self.category_items = items_df.groupby('category')['item_id'].apply(list).to_dict()
        
        # 计算物品相似度（基于类别）
        item_similarity = {}
        for item_id in items_df['item_id']:
            item_category = items_df[items_df['item_id'] == item_id]['category'].iloc[0]
            similar_items = []
            
            for other_item in items_df['item_id']:
                if other_item != item_id:
                    other_category = items_df[items_df['item_id'] == other_item]['category'].iloc[0]
                    similarity = 1.0 if item_category == other_category else 0.0
                    similar_items.append((other_item, similarity))
            
            item_similarity[item_id] = dict(similar_items)
        
        self.item_similarity = pd.DataFrame(item_similarity).fillna(0)
        
        self.logger.info("基于类别的推荐模型训练完成")
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """基于用户历史评分推荐相似物品"""
        # 获取用户历史评分的物品
        user_items = self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'].tolist()
        
        if not user_items:
            return []
        
        # 计算用户对所有物品的相似度分数
        item_scores = {}
        for item_id in self.item_similarity.index:
            if item_id not in user_items:
                score = 0
                for user_item in user_items:
                    if user_item in self.item_similarity.columns:
                        score += self.item_similarity.loc[item_id, user_item]
                item_scores[item_id] = score / len(user_items)
        
        # 按分数排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'item_id': item_id,
                'score': score,
                'algorithm': 'category_content_based'
            }
            for item_id, score in sorted_items[:n_recommendations]
        ]

class PopularityBasedContent(ContentBasedBase):
    """基于流行度的内容推荐"""
    
    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """训练基于流行度的推荐模型"""
        self.ratings_df = ratings_df
        
        # 计算物品流行度
        item_popularity = ratings_df.groupby('item_id').agg({
            'rating': ['count', 'mean']
        }).round(3)
        item_popularity.columns = ['rating_count', 'avg_rating']
        
        # 计算流行度分数（评分数量 * 平均评分）
        item_popularity['popularity_score'] = (
            item_popularity['rating_count'] * item_popularity['avg_rating']
        )
        
        self.item_popularity = item_popularity.sort_values('popularity_score', ascending=False)
        
        self.logger.info("基于流行度的推荐模型训练完成")
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """基于流行度推荐物品"""
        # 获取用户历史评分的物品
        user_items = self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'].tolist()
        
        # 过滤掉用户已经评分的物品
        available_items = self.item_popularity[~self.item_popularity.index.isin(user_items)]
        
        return [
            {
                'item_id': item_id,
                'score': row['popularity_score'],
                'rating_count': row['rating_count'],
                'avg_rating': row['avg_rating'],
                'algorithm': 'popularity_content_based'
            }
            for item_id, row in available_items.head(n_recommendations).iterrows()
        ]