"""
推荐系统配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """基础配置类"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # 数据库配置
    DB_HOST = os.environ.get('DB_HOST') or 'localhost'
    DB_PORT = int(os.environ.get('DB_PORT') or 3306)
    DB_USER = os.environ.get('DB_USER') or 'root'
    DB_PASSWORD = os.environ.get('DB_PASSWORD') or ''
    DB_NAME = os.environ.get('DB_NAME') or 'recommendation_system'

    # 推荐系统配置
    RECOMMENDATION_ALGORITHMS = {
        'collaborative_filtering': True,
        'content_based': True,
        'matrix_factorization': True,
        'deep_learning': True
    }

    # 模型参数
    MODEL_PARAMS = {
        'n_factors': 50,
        'n_epochs': 20,
        'lr_all': 0.005,
        'reg_all': 0.02,
        'min_rating': 1,
        'max_rating': 5,
        'recency_half_life_days': int(os.environ.get('RECENCY_HALF_LIFE_DAYS') or 180)
    }

    # 数据来源权重
    SOURCE_WEIGHTS = {
        'dataset_order': float(os.environ.get('WEIGHT_DATASET_ORDER') or 1.0),
        'api_order': float(os.environ.get('WEIGHT_API_ORDER') or 0.8)
    }

    # 推荐数量
    DEFAULT_RECOMMENDATIONS = int(os.environ.get('DEFAULT_RECOMMENDATIONS') or 10)
    MAX_RECOMMENDATIONS = int(os.environ.get('MAX_RECOMMENDATIONS') or 50)

    # 数据路径
    DATA_DIR = os.environ.get('DATA_DIR') or 'data'
    MODEL_DIR = os.environ.get('MODEL_DIR') or 'models'
    LOG_DIR = os.environ.get('LOG_DIR') or 'logs'

    # API配置
    API_HOST = os.environ.get('API_HOST') or '0.0.0.0'
    API_PORT = int(os.environ.get('API_PORT') or 5000)
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
