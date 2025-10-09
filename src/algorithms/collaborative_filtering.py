"""
协同过滤推荐算法
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import logging
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

class CollaborativeFilteringBase(ABC):
    """协同过滤基类"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
    
    @abstractmethod
    def fit(self, user_item_matrix: pd.DataFrame):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """为用户推荐物品"""
        pass

class UserBasedCF(CollaborativeFilteringBase):
    """基于用户的协同过滤"""
    
    def fit(self, user_item_matrix: pd.DataFrame):
        """训练基于用户的协同过滤模型"""
        self.user_item_matrix = user_item_matrix.fillna(0)
        
        # 计算用户相似度矩阵
        self.user_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix.values),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        self.logger.info("基于用户的协同过滤模型训练完成")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return 0.0
        
        # 获取用户评分
        user_ratings = self.user_item_matrix.loc[user_id]
        if user_ratings[item_id] != 0:  # 用户已经评分过
            return user_ratings[item_id]
        
        # 找到相似用户
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:11]  # 排除自己
        
        # 计算加权平均评分
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user, similarity in similar_users.items():
            if similarity > 0 and self.user_item_matrix.loc[similar_user, item_id] != 0:
                weighted_sum += similarity * self.user_item_matrix.loc[similar_user, item_id]
                similarity_sum += abs(similarity)
        
        if similarity_sum == 0:
            return 0.0
        
        return weighted_sum / similarity_sum
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """为用户推荐物品"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            if pred_rating > 0:
                predictions.append({
                    'item_id': item_id,
                    'predicted_rating': pred_rating,
                    'algorithm': 'user_based_cf'
                })
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:n_recommendations]

class ItemBasedCF(CollaborativeFilteringBase):
    """基于物品的协同过滤"""
    
    def fit(self, user_item_matrix: pd.DataFrame):
        """训练基于物品的协同过滤模型"""
        self.user_item_matrix = user_item_matrix.fillna(0)
        
        # 计算物品相似度矩阵
        self.item_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix.T.values),
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        self.logger.info("基于物品的协同过滤模型训练完成")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return 0.0
        
        # 获取用户评分
        user_ratings = self.user_item_matrix.loc[user_id]
        if user_ratings[item_id] != 0:  # 用户已经评分过
            return user_ratings[item_id]
        
        # 找到相似物品
        similar_items = self.item_similarity[item_id].sort_values(ascending=False)[1:11]  # 排除自己
        
        # 计算加权平均评分
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_item, similarity in similar_items.items():
            if similarity > 0 and user_ratings[similar_item] != 0:
                weighted_sum += similarity * user_ratings[similar_item]
                similarity_sum += abs(similarity)
        
        if similarity_sum == 0:
            return 0.0
        
        return weighted_sum / similarity_sum
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """为用户推荐物品"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            if pred_rating > 0:
                predictions.append({
                    'item_id': item_id,
                    'predicted_rating': pred_rating,
                    'algorithm': 'item_based_cf'
                })
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:n_recommendations]

class MatrixFactorizationCF(CollaborativeFilteringBase):
    """矩阵分解协同过滤"""
    
    def __init__(self, config):
        super().__init__(config)
        self.n_factors = config['MODEL_PARAMS']['n_factors']
        self.n_epochs = config['MODEL_PARAMS']['n_epochs']
        self.lr = config['MODEL_PARAMS']['lr_all']
        self.reg = config['MODEL_PARAMS']['reg_all']
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, user_item_matrix: pd.DataFrame):
        """训练矩阵分解模型"""
        self.user_item_matrix = user_item_matrix.fillna(0)
        
        # 获取用户和物品数量
        n_users, n_items = self.user_item_matrix.shape
        
        # 初始化用户和物品因子矩阵
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # 获取非零评分的索引
        user_indices, item_indices = np.where(self.user_item_matrix.values > 0)
        
        # 随机梯度下降训练
        for epoch in range(self.n_epochs):
            for i, j in zip(user_indices, item_indices):
                rating = self.user_item_matrix.iloc[i, j]
                prediction = np.dot(self.user_factors[i], self.item_factors[j])
                error = rating - prediction
                
                # 更新因子
                user_factor_i = self.user_factors[i].copy()
                self.user_factors[i] += self.lr * (error * self.item_factors[j] - self.reg * self.user_factors[i])
                self.item_factors[j] += self.lr * (error * user_factor_i - self.reg * self.item_factors[j])
            
            if epoch % 5 == 0:
                self.logger.info(f"矩阵分解训练进度: {epoch}/{self.n_epochs}")
        
        self.logger.info("矩阵分解协同过滤模型训练完成")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return 0.0
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return max(0, min(5, prediction))  # 限制在0-5范围内
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """为用户推荐物品"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            if pred_rating > 0:
                predictions.append({
                    'item_id': item_id,
                    'predicted_rating': pred_rating,
                    'algorithm': 'matrix_factorization_cf'
                })
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:n_recommendations]

class SVDCF(CollaborativeFilteringBase):
    """SVD协同过滤"""
    
    def __init__(self, config):
        super().__init__(config)
        self.n_components = config['MODEL_PARAMS']['n_factors']
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
    
    def fit(self, user_item_matrix: pd.DataFrame):
        """训练SVD模型"""
        self.user_item_matrix = user_item_matrix.fillna(0)
        
        # 应用SVD
        self.svd.fit(self.user_item_matrix.values)
        
        # 获取用户和物品的潜在因子
        self.user_factors = self.svd.transform(self.user_item_matrix.values)
        self.item_factors = self.svd.components_.T
        
        self.logger.info("SVD协同过滤模型训练完成")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return 0.0
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return max(0, min(5, prediction))  # 限制在0-5范围内
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """为用户推荐物品"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            if pred_rating > 0:
                predictions.append({
                    'item_id': item_id,
                    'predicted_rating': pred_rating,
                    'algorithm': 'svd_cf'
                })
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:n_recommendations]