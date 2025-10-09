"""
推荐系统评估模块
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationEvaluator:
    """推荐系统评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_rating_prediction(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """评估评分预测性能"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 移除无效预测
        valid_mask = ~(np.isnan(y_pred) | np.isinf(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'r2': -float('inf'),
                'coverage': 0.0
            }
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'coverage': len(y_true) / len(y_pred) if len(y_pred) > 0 else 0
        }
    
    def evaluate_ranking(self, recommendations: List[Dict], ground_truth: List[int]) -> Dict[str, float]:
        """评估排序性能"""
        if not recommendations or not ground_truth:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ndcg': 0.0,
                'map': 0.0
            }
        
        # 提取推荐物品ID
        recommended_items = [rec['item_id'] for rec in recommendations]
        
        # 计算精确率
        precision = len(set(recommended_items) & set(ground_truth)) / len(recommended_items)
        
        # 计算召回率
        recall = len(set(recommended_items) & set(ground_truth)) / len(ground_truth)
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算NDCG
        ndcg = self._calculate_ndcg(recommended_items, ground_truth)
        
        # 计算MAP
        map_score = self._calculate_map(recommended_items, ground_truth)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ndcg': ndcg,
            'map': map_score
        }
    
    def evaluate_diversity(self, recommendations: List[Dict], item_features: pd.DataFrame = None) -> Dict[str, float]:
        """评估推荐多样性"""
        if not recommendations:
            return {'diversity': 0.0, 'intra_list_diversity': 0.0}
        
        recommended_items = [rec['item_id'] for rec in recommendations]
        
        # 基于物品特征的多样性
        if item_features is not None:
            diversity = self._calculate_content_diversity(recommended_items, item_features)
        else:
            diversity = 0.0
        
        # 基于评分的多样性
        scores = [rec.get('predicted_rating', rec.get('score', 0)) for rec in recommendations]
        intra_list_diversity = np.std(scores) if len(scores) > 1 else 0.0
        
        return {
            'diversity': diversity,
            'intra_list_diversity': intra_list_diversity
        }
    
    def evaluate_novelty(self, recommendations: List[Dict], popular_items: List[int]) -> Dict[str, float]:
        """评估推荐新颖性"""
        if not recommendations:
            return {'novelty': 0.0, 'unexpectedness': 0.0}
        
        recommended_items = [rec['item_id'] for rec in recommendations]
        
        # 计算非热门物品比例
        non_popular_count = len([item for item in recommended_items if item not in popular_items])
        novelty = non_popular_count / len(recommended_items)
        
        # 计算意外性（基于评分分布）
        scores = [rec.get('predicted_rating', rec.get('score', 0)) for rec in recommendations]
        if scores:
            score_std = np.std(scores)
            unexpectedness = min(score_std / 5.0, 1.0)  # 标准化到0-1
        else:
            unexpectedness = 0.0
        
        return {
            'novelty': novelty,
            'unexpectedness': unexpectedness
        }
    
    def evaluate_coverage(self, all_recommendations: Dict[int, List[Dict]], all_items: List[int]) -> Dict[str, float]:
        """评估推荐覆盖率"""
        if not all_recommendations or not all_items:
            return {'catalog_coverage': 0.0, 'user_coverage': 0.0}
        
        # 计算目录覆盖率
        recommended_items = set()
        for user_recs in all_recommendations.values():
            for rec in user_recs:
                recommended_items.add(rec['item_id'])
        
        catalog_coverage = len(recommended_items) / len(all_items)
        
        # 计算用户覆盖率
        users_with_recs = len([user for user, recs in all_recommendations.items() if recs])
        user_coverage = users_with_recs / len(all_recommendations)
        
        return {
            'catalog_coverage': catalog_coverage,
            'user_coverage': user_coverage
        }
    
    def cross_validation_evaluate(self, ratings_df: pd.DataFrame, model, n_folds: int = 5) -> Dict[str, float]:
        """交叉验证评估"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        all_metrics = []
        
        for train_idx, test_idx in kf.split(ratings_df):
            train_df = ratings_df.iloc[train_idx]
            test_df = ratings_df.iloc[test_idx]
            
            # 训练模型
            user_item_matrix = train_df.pivot_table(
                index='user_id', 
                columns='item_id', 
                values='rating', 
                fill_value=0
            )
            
            if hasattr(model, 'fit'):
                model.fit(user_item_matrix)
            
            # 评估
            y_true = []
            y_pred = []
            
            for _, row in test_df.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                actual_rating = row['rating']
                
                try:
                    if hasattr(model, 'predict'):
                        pred_rating = model.predict(user_id, item_id)
                        y_true.append(actual_rating)
                        y_pred.append(pred_rating)
                except:
                    continue
            
            if y_true and y_pred:
                metrics = self.evaluate_rating_prediction(y_true, y_pred)
                all_metrics.append(metrics)
        
        # 计算平均指标
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            return avg_metrics
        else:
            return {'rmse': float('inf'), 'mae': float('inf'), 'coverage': 0.0}
    
    def _calculate_ndcg(self, recommended_items: List[int], ground_truth: List[int], k: int = None) -> float:
        """计算NDCG"""
        if k is None:
            k = len(recommended_items)
        
        # 创建相关性标签
        relevance = [1 if item in ground_truth else 0 for item in recommended_items[:k]]
        
        # 计算DCG
        dcg = sum(relevance[i] / np.log2(i + 2) for i in range(len(relevance)))
        
        # 计算IDCG
        ideal_relevance = [1] * min(len(ground_truth), k)
        idcg = sum(ideal_relevance[i] / np.log2(i + 2) for i in range(len(ideal_relevance)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_map(self, recommended_items: List[int], ground_truth: List[int]) -> float:
        """计算MAP"""
        if not ground_truth:
            return 0.0
        
        precision_at_k = []
        for k in range(1, len(recommended_items) + 1):
            relevant_items = set(ground_truth)
            recommended_k = recommended_items[:k]
            relevant_recommended = len(set(recommended_k) & relevant_items)
            precision_at_k.append(relevant_recommended / k)
        
        # 计算平均精度
        map_score = np.mean(precision_at_k)
        return map_score
    
    def _calculate_content_diversity(self, recommended_items: List[int], item_features: pd.DataFrame) -> float:
        """计算基于内容的多样性"""
        if len(recommended_items) < 2:
            return 0.0
        
        # 获取推荐物品的特征
        rec_features = item_features.loc[recommended_items]
        
        # 计算特征相似度矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(rec_features.values)
        
        # 计算平均相似度
        n = len(similarity_matrix)
        avg_similarity = (np.sum(similarity_matrix) - n) / (n * (n - 1))
        
        # 多样性 = 1 - 平均相似度
        diversity = 1 - avg_similarity
        return max(0, diversity)
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """生成评估报告"""
        report = "# 推荐系统评估报告\n\n"
        
        for model_name, metrics in evaluation_results.items():
            report += f"## {model_name}\n\n"
            
            if 'rmse' in metrics:
                report += "### 评分预测性能\n"
                report += f"- RMSE: {metrics['rmse']:.4f}\n"
                report += f"- MAE: {metrics['mae']:.4f}\n"
                report += f"- MAPE: {metrics['mape']:.4f}%\n"
                report += f"- R²: {metrics['r2']:.4f}\n"
                report += f"- 覆盖率: {metrics['coverage']:.4f}\n\n"
            
            if 'precision' in metrics:
                report += "### 排序性能\n"
                report += f"- 精确率: {metrics['precision']:.4f}\n"
                report += f"- 召回率: {metrics['recall']:.4f}\n"
                report += f"- F1分数: {metrics['f1']:.4f}\n"
                report += f"- NDCG: {metrics['ndcg']:.4f}\n"
                report += f"- MAP: {metrics['map']:.4f}\n\n"
            
            if 'diversity' in metrics:
                report += "### 多样性\n"
                report += f"- 内容多样性: {metrics['diversity']:.4f}\n"
                report += f"- 评分多样性: {metrics['intra_list_diversity']:.4f}\n\n"
            
            if 'novelty' in metrics:
                report += "### 新颖性\n"
                report += f"- 新颖性: {metrics['novelty']:.4f}\n"
                report += f"- 意外性: {metrics['unexpectedness']:.4f}\n\n"
        
        return report
    
    def plot_evaluation_results(self, evaluation_results: Dict[str, Dict[str, float]], 
                              save_path: str = None):
        """绘制评估结果图表"""
        # 准备数据
        models = list(evaluation_results.keys())
        metrics = ['rmse', 'mae', 'precision', 'recall', 'f1', 'ndcg']
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = []
            for model in models:
                if metric in evaluation_results[model]:
                    values.append(evaluation_results[model][metric])
                else:
                    values.append(0)
            
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()