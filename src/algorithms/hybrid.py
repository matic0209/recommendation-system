"""
混合推荐算法
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import RandomForestRegressor
    LIGHTGBM_AVAILABLE = False

class HybridRecommendation:
    """混合推荐系统"""
    
    def __init__(self, config, collaborative_models, content_models):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.collaborative_models = collaborative_models
        self.content_models = content_models
        self.weights = {}
        self.meta_model = None
    
    def set_weights(self, weights: Dict[str, float]):
        """设置各算法的权重"""
        self.weights = weights
        self.logger.info(f"设置算法权重: {weights}")
    
    def weighted_ensemble(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """加权集成推荐"""
        all_recommendations = {}
        
        # 收集所有算法的推荐结果
        for name, model in self.collaborative_models.items():
            try:
                recommendations = model.recommend(user_id, n_recommendations * 2)
                weight = self.weights.get(name, 1.0)
                
                for rec in recommendations:
                    item_id = rec['item_id']
                    score = rec.get('predicted_rating', rec.get('score', 0))
                    
                    if item_id not in all_recommendations:
                        all_recommendations[item_id] = {
                            'item_id': item_id,
                            'scores': {},
                            'total_score': 0
                        }
                    
                    all_recommendations[item_id]['scores'][name] = score * weight
                    all_recommendations[item_id]['total_score'] += score * weight
            
            except Exception as e:
                self.logger.warning(f"协同过滤算法 {name} 推荐失败: {str(e)}")
        
        for name, model in self.content_models.items():
            try:
                recommendations = model.recommend(user_id, n_recommendations * 2)
                weight = self.weights.get(name, 1.0)
                
                for rec in recommendations:
                    item_id = rec['item_id']
                    score = rec.get('predicted_rating', rec.get('score', 0))
                    
                    if item_id not in all_recommendations:
                        all_recommendations[item_id] = {
                            'item_id': item_id,
                            'scores': {},
                            'total_score': 0
                        }
                    
                    all_recommendations[item_id]['scores'][name] = score * weight
                    all_recommendations[item_id]['total_score'] += score * weight
            
            except Exception as e:
                self.logger.warning(f"内容推荐算法 {name} 推荐失败: {str(e)}")
        
        # 按总分排序
        sorted_recommendations = sorted(
            all_recommendations.values(),
            key=lambda x: x['total_score'],
            reverse=True
        )
        
        return [
            {
                'item_id': rec['item_id'],
                'total_score': rec['total_score'],
                'algorithm_scores': rec['scores'],
                'algorithm': 'weighted_ensemble'
            }
            for rec in sorted_recommendations[:n_recommendations]
        ]
    
    def train_meta_model(self, ratings_df: pd.DataFrame, user_item_matrix: pd.DataFrame):
        """训练元学习模型"""
        self.logger.info("开始训练元学习模型")
        
        # 准备训练数据
        X = []
        y = []
        
        for _, rating in ratings_df.iterrows():
            user_id = rating['user_id']
            item_id = rating['item_id']
            actual_rating = rating['rating']
            
            # 获取各算法的预测
            features = []
            
            # 协同过滤特征
            for name, model in self.collaborative_models.items():
                try:
                    pred = model.predict(user_id, item_id)
                    features.append(pred)
                except:
                    features.append(0)
            
            # 内容推荐特征
            for name, model in self.content_models.items():
                try:
                    # 对于内容推荐，我们需要计算相似度分数
                    if hasattr(model, 'item_similarity') and item_id in model.item_similarity.index:
                        user_items = ratings_df[ratings_df['user_id'] == user_id]['item_id'].tolist()
                        if user_items:
                            score = 0
                            for user_item in user_items:
                                if user_item in model.item_similarity.columns:
                                    score += model.item_similarity.loc[item_id, user_item]
                            features.append(score / len(user_items))
                        else:
                            features.append(0)
                    else:
                        features.append(0)
                except:
                    features.append(0)
            
            # 用户和物品特征
            if user_id in user_item_matrix.index:
                user_ratings = user_item_matrix.loc[user_id]
                features.extend([
                    user_ratings.mean(),
                    user_ratings.std(),
                    (user_ratings > 0).sum()
                ])
            else:
                features.extend([0, 0, 0])
            
            if item_id in user_item_matrix.columns:
                item_ratings = user_item_matrix[item_id]
                features.extend([
                    item_ratings.mean(),
                    item_ratings.std(),
                    (item_ratings > 0).sum()
                ])
            else:
                features.extend([0, 0, 0])
            
            X.append(features)
            y.append(actual_rating)
        
        # 训练元模型
        X = np.array(X)
        y = np.array(y)
        
        if LIGHTGBM_AVAILABLE:
            self.meta_model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
        else:
            from sklearn.ensemble import RandomForestRegressor  # local import fallback
            self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)

        self.meta_model.fit(X, y)
        
        self.logger.info("元学习模型训练完成")
    
    def meta_learning_recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """使用元学习模型推荐"""
        if self.meta_model is None:
            self.logger.warning("元学习模型未训练，使用加权集成")
            return self.weighted_ensemble(user_id, n_recommendations)
        
        # 获取所有候选物品
        all_items = set()
        for model in list(self.collaborative_models.values()) + list(self.content_models.values()):
            if hasattr(model, 'user_item_matrix'):
                all_items.update(model.user_item_matrix.columns)
        
        # 获取用户已评分的物品
        user_rated_items = set()
        for model in self.collaborative_models.values():
            if hasattr(model, 'user_item_matrix') and user_id in model.user_item_matrix.index:
                user_rated_items.update(
                    model.user_item_matrix.loc[user_id][model.user_item_matrix.loc[user_id] > 0].index
                )
        
        # 候选物品
        candidate_items = all_items - user_rated_items
        
        predictions = []
        for item_id in candidate_items:
            # 准备特征
            features = []
            
            # 协同过滤特征
            for model in self.collaborative_models.values():
                try:
                    pred = model.predict(user_id, item_id)
                    features.append(pred)
                except:
                    features.append(0)
            
            # 内容推荐特征
            for model in self.content_models.values():
                try:
                    if hasattr(model, 'item_similarity') and item_id in model.item_similarity.index:
                        user_items = list(user_rated_items)
                        if user_items:
                            score = 0
                            for user_item in user_items:
                                if user_item in model.item_similarity.columns:
                                    score += model.item_similarity.loc[item_id, user_item]
                            features.append(score / len(user_items))
                        else:
                            features.append(0)
                    else:
                        features.append(0)
                except:
                    features.append(0)
            
            # 用户和物品特征
            user_features = [0, 0, 0]  # 默认值
            item_features = [0, 0, 0]  # 默认值
            
            for model in self.collaborative_models.values():
                if hasattr(model, 'user_item_matrix'):
                    if user_id in model.user_item_matrix.index:
                        user_ratings = model.user_item_matrix.loc[user_id]
                        user_features = [
                            user_ratings.mean(),
                            user_ratings.std(),
                            (user_ratings > 0).sum()
                        ]
                        break
            
            for model in self.collaborative_models.values():
                if hasattr(model, 'user_item_matrix'):
                    if item_id in model.user_item_matrix.columns:
                        item_ratings = model.user_item_matrix[item_id]
                        item_features = [
                            item_ratings.mean(),
                            item_ratings.std(),
                            (item_ratings > 0).sum()
                        ]
                        break
            
            features.extend(user_features)
            features.extend(item_features)
            
            # 预测评分
            pred_rating = self.meta_model.predict([features])[0]
            
            predictions.append({
                'item_id': item_id,
                'predicted_rating': pred_rating,
                'algorithm': 'meta_learning'
            })
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:n_recommendations]
    
    def switch_algorithm(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """算法切换策略"""
        # 获取用户评分数量
        user_rating_count = 0
        for model in self.collaborative_models.values():
            if hasattr(model, 'user_item_matrix') and user_id in model.user_item_matrix.index:
                user_rating_count = (model.user_item_matrix.loc[user_id] > 0).sum()
                break
        
        # 根据用户评分数量选择算法
        if user_rating_count < 5:
            # 冷启动用户，使用内容推荐
            self.logger.info(f"用户 {user_id} 评分数量少，使用内容推荐")
            for model in self.content_models.values():
                try:
                    return model.recommend(user_id, n_recommendations)
                except:
                    continue
        else:
            # 活跃用户，使用协同过滤
            self.logger.info(f"用户 {user_id} 评分数量充足，使用协同过滤")
            for model in self.collaborative_models.values():
                try:
                    return model.recommend(user_id, n_recommendations)
                except:
                    continue
        
        # 如果所有算法都失败，返回空列表
        return []
