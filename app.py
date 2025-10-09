"""
推荐系统主应用
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

from config import config
from src.database import DatabaseManager
from src.recommendation_engine import RecommendationEngine
from src.data_processor import DataProcessor

def create_app(config_name='default'):
    """创建Flask应用"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # 启用CORS
    CORS(app)
    
    # 配置日志
    setup_logging(app)
    
    # 初始化组件
    db_manager = DatabaseManager(app.config)
    data_processor = DataProcessor(app.config)
    rec_engine = RecommendationEngine(app.config, db_manager, data_processor)
    
    @app.route('/')
    def index():
        """首页"""
        return jsonify({
            'message': '推荐系统API',
            'version': '1.0.0',
            'status': 'running'
        })
    
    @app.route('/health')
    def health():
        """健康检查"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/recommend/<int:user_id>', methods=['GET'])
    def get_recommendations(user_id):
        """获取用户推荐"""
        try:
            algorithm = request.args.get('algorithm', 'collaborative_filtering')
            n_recommendations = int(request.args.get('n', app.config['DEFAULT_RECOMMENDATIONS']))
            
            recommendations = rec_engine.get_recommendations(
                user_id=user_id,
                algorithm=algorithm,
                n_recommendations=n_recommendations
            )
            
            return jsonify({
                'user_id': user_id,
                'algorithm': algorithm,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            app.logger.error(f"推荐生成失败: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/similar_items/<int:item_id>', methods=['GET'])
    def get_similar_items(item_id):
        """获取相似物品"""
        try:
            n_similar = int(request.args.get('n', 10))
            
            similar_items = rec_engine.get_similar_items(
                item_id=item_id,
                n_similar=n_similar
            )
            
            return jsonify({
                'item_id': item_id,
                'similar_items': similar_items,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            app.logger.error(f"相似物品获取失败: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/train', methods=['POST'])
    def train_models():
        """训练推荐模型"""
        try:
            algorithm = request.json.get('algorithm', 'all')
            
            result = rec_engine.train_models(algorithm=algorithm)
            
            return jsonify({
                'message': '模型训练完成',
                'algorithm': algorithm,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            app.logger.error(f"模型训练失败: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/evaluate', methods=['POST'])
    def evaluate_models():
        """评估推荐模型"""
        try:
            algorithm = request.json.get('algorithm', 'all')
            test_size = request.json.get('test_size', 0.2)
            
            evaluation_results = rec_engine.evaluate_models(
                algorithm=algorithm,
                test_size=test_size
            )
            
            return jsonify({
                'evaluation_results': evaluation_results,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            app.logger.error(f"模型评估失败: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/stats', methods=['GET'])
    def get_stats():
        """获取系统统计信息"""
        try:
            stats = rec_engine.get_system_stats()
            return jsonify(stats)
            
        except Exception as e:
            app.logger.error(f"统计信息获取失败: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return app

def setup_logging(app):
    """设置日志"""
    if not os.path.exists(app.config['LOG_DIR']):
        os.makedirs(app.config['LOG_DIR'])
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[
            logging.FileHandler(f"{app.config['LOG_DIR']}/app.log"),
            logging.StreamHandler()
        ]
    )

if __name__ == '__main__':
    app = create_app()
    app.run(
        host=app.config['API_HOST'],
        port=app.config['API_PORT'],
        debug=app.config['DEBUG']
    )