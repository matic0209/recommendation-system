"""
推荐引擎核心模块
"""
import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .database import DatabaseManager
from .data_processor import DataProcessor
from .algorithms.collaborative_filtering import (
    UserBasedCF, ItemBasedCF, MatrixFactorizationCF, SVDCF
)
from .algorithms.content_based import (
    TFIDFContentBased, HybridContentBased, CategoryBasedContent, PopularityBasedContent
)
from .algorithms.hybrid import HybridRecommendation

class RecommendationEngine:
    """推荐引擎"""
    
    def __init__(self, config, db_manager: DatabaseManager, data_processor: DataProcessor):
        self.config = config
        self.db_manager = db_manager
        self.data_processor = data_processor
        self.logger = logging.getLogger(__name__)
        
        # 初始化模型
        self.models = {}
        self.hybrid_model = None
        
        # 确保模型目录存在
        os.makedirs(config['MODEL_DIR'], exist_ok=True)
        
        # 加载已训练的模型
        self._load_models()
    
    def _load_models(self):
        """加载已训练的模型"""
        model_files = {
            'user_based_cf': 'user_based_cf.pkl',
            'item_based_cf': 'item_based_cf.pkl',
            'matrix_factorization_cf': 'matrix_factorization_cf.pkl',
            'svd_cf': 'svd_cf.pkl',
            'tfidf_content': 'tfidf_content.pkl',
            'hybrid_content': 'hybrid_content.pkl',
            'category_content': 'category_content.pkl',
            'popularity_content': 'popularity_content.pkl',
            'hybrid_model': 'hybrid_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.config['MODEL_DIR'], filename)
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    self.logger.info(f"成功加载模型: {model_name}")
                except Exception as e:
                    self.logger.warning(f"加载模型 {model_name} 失败: {str(e)}")
    
    def _save_model(self, model, model_name: str):
        """保存模型"""
        model_path = os.path.join(self.config['MODEL_DIR'], f"{model_name}.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"成功保存模型: {model_name}")
        except Exception as e:
            self.logger.error(f"保存模型 {model_name} 失败: {str(e)}")
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """准备训练数据"""
        self.logger.info("开始准备训练数据")
        
        # 获取原始数据
        ratings_df = self.db_manager.get_user_ratings()
        items_df = self.db_manager.get_item_features()
        users_df = self.db_manager.get_user_features()
        
        # 预处理数据
        ratings_df = self.data_processor.preprocess_ratings(ratings_df)
        items_df = self.data_processor.preprocess_items(items_df)
        users_df = self.data_processor.preprocess_users(users_df)
        
        # 创建用户-物品矩阵
        user_item_matrix = self.data_processor.create_user_item_matrix(ratings_df)
        
        self.logger.info("数据准备完成")
        return ratings_df, items_df, users_df, user_item_matrix
    
    def train_collaborative_filtering(self, user_item_matrix: pd.DataFrame):
        """训练协同过滤模型"""
        self.logger.info("开始训练协同过滤模型")
        
        # 用户协同过滤
        if self.config['RECOMMENDATION_ALGORITHMS']['collaborative_filtering']:
            try:
                user_cf = UserBasedCF(self.config)
                user_cf.fit(user_item_matrix)
                self.models['user_based_cf'] = user_cf
                self._save_model(user_cf, 'user_based_cf')
                self.logger.info("用户协同过滤模型训练完成")
            except Exception as e:
                self.logger.error(f"用户协同过滤模型训练失败: {str(e)}")
            
            try:
                item_cf = ItemBasedCF(self.config)
                item_cf.fit(user_item_matrix)
                self.models['item_based_cf'] = item_cf
                self._save_model(item_cf, 'item_based_cf')
                self.logger.info("物品协同过滤模型训练完成")
            except Exception as e:
                self.logger.error(f"物品协同过滤模型训练失败: {str(e)}")
            
            try:
                mf_cf = MatrixFactorizationCF(self.config)
                mf_cf.fit(user_item_matrix)
                self.models['matrix_factorization_cf'] = mf_cf
                self._save_model(mf_cf, 'matrix_factorization_cf')
                self.logger.info("矩阵分解协同过滤模型训练完成")
            except Exception as e:
                self.logger.error(f"矩阵分解协同过滤模型训练失败: {str(e)}")
            
            try:
                svd_cf = SVDCF(self.config)
                svd_cf.fit(user_item_matrix)
                self.models['svd_cf'] = svd_cf
                self._save_model(svd_cf, 'svd_cf')
                self.logger.info("SVD协同过滤模型训练完成")
            except Exception as e:
                self.logger.error(f"SVD协同过滤模型训练失败: {str(e)}")
    
    def train_content_based(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """训练基于内容的推荐模型"""
        self.logger.info("开始训练基于内容的推荐模型")
        
        if self.config['RECOMMENDATION_ALGORITHMS']['content_based']:
            try:
                tfidf_model = TFIDFContentBased(self.config)
                tfidf_model.fit(items_df, ratings_df)
                self.models['tfidf_content'] = tfidf_model
                self._save_model(tfidf_model, 'tfidf_content')
                self.logger.info("TF-IDF内容推荐模型训练完成")
            except Exception as e:
                self.logger.error(f"TF-IDF内容推荐模型训练失败: {str(e)}")
            
            try:
                hybrid_content = HybridContentBased(self.config)
                hybrid_content.fit(items_df, ratings_df)
                self.models['hybrid_content'] = hybrid_content
                self._save_model(hybrid_content, 'hybrid_content')
                self.logger.info("混合内容推荐模型训练完成")
            except Exception as e:
                self.logger.error(f"混合内容推荐模型训练失败: {str(e)}")
            
            try:
                category_content = CategoryBasedContent(self.config)
                category_content.fit(items_df, ratings_df)
                self.models['category_content'] = category_content
                self._save_model(category_content, 'category_content')
                self.logger.info("基于类别的内容推荐模型训练完成")
            except Exception as e:
                self.logger.error(f"基于类别的内容推荐模型训练失败: {str(e)}")
            
            try:
                popularity_content = PopularityBasedContent(self.config)
                popularity_content.fit(items_df, ratings_df)
                self.models['popularity_content'] = popularity_content
                self._save_model(popularity_content, 'popularity_content')
                self.logger.info("基于流行度的内容推荐模型训练完成")
            except Exception as e:
                self.logger.error(f"基于流行度的内容推荐模型训练失败: {str(e)}")
    
    def train_hybrid_model(self, ratings_df: pd.DataFrame, user_item_matrix: pd.DataFrame):
        """训练混合推荐模型"""
        self.logger.info("开始训练混合推荐模型")
        
        try:
            # 收集协同过滤模型
            cf_models = {}
            for name, model in self.models.items():
                if 'cf' in name:
                    cf_models[name] = model
            
            # 收集内容推荐模型
            content_models = {}
            for name, model in self.models.items():
                if 'content' in name:
                    content_models[name] = model
            
            # 创建混合模型
            self.hybrid_model = HybridRecommendation(
                self.config, cf_models, content_models
            )
            
            # 设置权重
            weights = {
                'user_based_cf': 0.3,
                'item_based_cf': 0.3,
                'matrix_factorization_cf': 0.2,
                'svd_cf': 0.2,
                'tfidf_content': 0.1,
                'hybrid_content': 0.1,
                'category_content': 0.1,
                'popularity_content': 0.1
            }
            self.hybrid_model.set_weights(weights)
            
            # 训练元学习模型
            self.hybrid_model.train_meta_model(ratings_df, user_item_matrix)
            
            self.models['hybrid_model'] = self.hybrid_model
            self._save_model(self.hybrid_model, 'hybrid_model')
            self.logger.info("混合推荐模型训练完成")
            
        except Exception as e:
            self.logger.error(f"混合推荐模型训练失败: {str(e)}")
    
    def train_models(self, algorithm: str = 'all'):
        """训练所有模型"""
        self.logger.info(f"开始训练推荐模型，算法: {algorithm}")
        
        # 准备数据
        ratings_df, items_df, users_df, user_item_matrix = self.prepare_data()
        
        # 训练协同过滤模型
        if algorithm in ['all', 'collaborative_filtering']:
            self.train_collaborative_filtering(user_item_matrix)
        
        # 训练内容推荐模型
        if algorithm in ['all', 'content_based']:
            self.train_content_based(items_df, ratings_df)
        
        # 训练混合模型
        if algorithm in ['all', 'hybrid']:
            self.train_hybrid_model(ratings_df, user_item_matrix)
        
        self.logger.info("所有模型训练完成")
        return {
            'status': 'success',
            'trained_models': list(self.models.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_recommendations(self, user_id: int, algorithm: str = 'collaborative_filtering', 
                          n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """获取推荐结果"""
        try:
            if algorithm == 'hybrid' and self.hybrid_model:
                return self.hybrid_model.meta_learning_recommend(user_id, n_recommendations)
            elif algorithm in self.models:
                model = self.models[algorithm]
                if hasattr(model, 'recommend'):
                    return model.recommend(user_id, n_recommendations)
                else:
                    self.logger.warning(f"模型 {algorithm} 没有 recommend 方法")
                    return []
            else:
                self.logger.warning(f"未找到算法 {algorithm}")
                return []
        except Exception as e:
            self.logger.error(f"获取推荐失败: {str(e)}")
            return []
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Dict[str, Any]]:
        """获取相似物品"""
        try:
            # 优先使用内容推荐模型
            for model_name, model in self.models.items():
                if 'content' in model_name and hasattr(model, 'get_similar_items'):
                    return model.get_similar_items(item_id, n_similar)
            
            # 如果没有内容推荐模型，使用协同过滤
            for model_name, model in self.models.items():
                if 'cf' in model_name and hasattr(model, 'item_similarity'):
                    if item_id in model.item_similarity.index:
                        similar_items = model.item_similarity[item_id].sort_values(ascending=False)[1:n_similar+1]
                        return [
                            {
                                'item_id': similar_item,
                                'similarity': similarity,
                                'algorithm': model_name
                            }
                            for similar_item, similarity in similar_items.items()
                        ]
            
            return []
        except Exception as e:
            self.logger.error(f"获取相似物品失败: {str(e)}")
            return []
    
    def evaluate_models(self, algorithm: str = 'all', test_size: float = 0.2) -> Dict[str, Any]:
        """评估模型性能"""
        self.logger.info(f"开始评估模型，算法: {algorithm}")
        
        # 准备数据
        ratings_df, items_df, users_df, user_item_matrix = self.prepare_data()
        
        # 分割训练和测试数据
        train_df, test_df = self.data_processor.split_data(ratings_df, test_size)
        
        evaluation_results = {}
        
        # 评估协同过滤模型
        if algorithm in ['all', 'collaborative_filtering']:
            cf_models = ['user_based_cf', 'item_based_cf', 'matrix_factorization_cf', 'svd_cf']
            for model_name in cf_models:
                if model_name in self.models:
                    try:
                        model = self.models[model_name]
                        metrics = self._evaluate_cf_model(model, test_df)
                        evaluation_results[model_name] = metrics
                    except Exception as e:
                        self.logger.error(f"评估模型 {model_name} 失败: {str(e)}")
        
        # 评估内容推荐模型
        if algorithm in ['all', 'content_based']:
            content_models = ['tfidf_content', 'hybrid_content', 'category_content', 'popularity_content']
            for model_name in content_models:
                if model_name in self.models:
                    try:
                        model = self.models[model_name]
                        metrics = self._evaluate_content_model(model, test_df)
                        evaluation_results[model_name] = metrics
                    except Exception as e:
                        self.logger.error(f"评估模型 {model_name} 失败: {str(e)}")
        
        # 评估混合模型
        if algorithm in ['all', 'hybrid'] and self.hybrid_model:
            try:
                metrics = self._evaluate_hybrid_model(self.hybrid_model, test_df)
                evaluation_results['hybrid_model'] = metrics
            except Exception as e:
                self.logger.error(f"评估混合模型失败: {str(e)}")
        
        self.logger.info("模型评估完成")
        return evaluation_results
    
    def _evaluate_cf_model(self, model, test_df: pd.DataFrame) -> Dict[str, float]:
        """评估协同过滤模型"""
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            try:
                pred_rating = model.predict(user_id, item_id)
                predictions.append(pred_rating)
                actuals.append(actual_rating)
            except:
                continue
        
        if not predictions:
            return {'rmse': float('inf'), 'mae': float('inf'), 'coverage': 0.0}
        
        # 计算指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        coverage = len(predictions) / len(test_df)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'n_predictions': len(predictions)
        }
    
    def _evaluate_content_model(self, model, test_df: pd.DataFrame) -> Dict[str, float]:
        """评估内容推荐模型"""
        # 内容推荐模型通常用于推荐，而不是预测评分
        # 这里我们计算推荐覆盖率
        user_recommendations = {}
        
        for user_id in test_df['user_id'].unique():
            try:
                recommendations = model.recommend(user_id, 10)
                user_recommendations[user_id] = [rec['item_id'] for rec in recommendations]
            except:
                user_recommendations[user_id] = []
        
        # 计算推荐覆盖率
        total_recommendations = sum(len(recs) for recs in user_recommendations.values())
        unique_recommendations = len(set(
            item_id for recs in user_recommendations.values() for item_id in recs
        ))
        
        coverage = unique_recommendations / total_recommendations if total_recommendations > 0 else 0
        
        return {
            'coverage': coverage,
            'total_recommendations': total_recommendations,
            'unique_recommendations': unique_recommendations
        }
    
    def _evaluate_hybrid_model(self, model, test_df: pd.DataFrame) -> Dict[str, float]:
        """评估混合模型"""
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            try:
                # 使用元学习模型预测
                recommendations = model.meta_learning_recommend(user_id, 100)
                pred_rating = 0
                for rec in recommendations:
                    if rec['item_id'] == item_id:
                        pred_rating = rec['predicted_rating']
                        break
                
                if pred_rating > 0:
                    predictions.append(pred_rating)
                    actuals.append(actual_rating)
            except:
                continue
        
        if not predictions:
            return {'rmse': float('inf'), 'mae': float('inf'), 'coverage': 0.0}
        
        # 计算指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        coverage = len(predictions) / len(test_df)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'n_predictions': len(predictions)
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            # 获取数据统计
            ratings_df = self.db_manager.get_user_ratings()
            items_df = self.db_manager.get_item_features()
            users_df = self.db_manager.get_user_features()
            
            stats = {
                'data_stats': {
                    'total_ratings': len(ratings_df),
                    'unique_users': ratings_df['user_id'].nunique() if not ratings_df.empty else 0,
                    'unique_items': ratings_df['item_id'].nunique() if not ratings_df.empty else 0,
                    'avg_rating': ratings_df['rating'].mean() if not ratings_df.empty else 0,
                    'total_items': len(items_df),
                    'total_users': len(users_df)
                },
                'model_stats': {
                    'trained_models': list(self.models.keys()),
                    'model_count': len(self.models),
                    'hybrid_model_available': self.hybrid_model is not None
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取系统统计信息失败: {str(e)}")
            return {'error': str(e)}