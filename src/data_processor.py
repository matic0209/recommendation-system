"""
数据处理和特征工程模块
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class DataProcessor:
    """数据处理器"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}

    # ------------------------------------------------------------------
    # 交互/订单数据预处理
    # ------------------------------------------------------------------
    def preprocess_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """将订单与调用记录转换为推荐模型可用的交互强度数据"""
        if df is None or df.empty:
            self.logger.warning("交互数据为空，无法构建行为矩阵")
            return pd.DataFrame(columns=["user_id", "item_id", "rating"])

        df = df.copy()

        df['price'] = pd.to_numeric(df.get('price'), errors='coerce').fillna(0).clip(lower=0)
        df['quantity'] = pd.to_numeric(df.get('quantity'), errors='coerce').fillna(1)
        df.loc[df['quantity'] <= 0, 'quantity'] = 1

        df['create_time'] = pd.to_datetime(df.get('create_time'), errors='coerce')
        df['event_time'] = pd.to_datetime(df.get('event_time'), errors='coerce')
        df['event_time'] = df['event_time'].fillna(df['create_time'])
        df['event_time'] = df['event_time'].fillna(pd.Timestamp.utcnow())

        half_life = self.config.get('MODEL_PARAMS', {}).get('recency_half_life_days', 180)
        now = pd.Timestamp.utcnow()
        if half_life and half_life > 0:
            df['days_since'] = (now - df['event_time']).dt.total_seconds() / 86400
            df['days_since'] = df['days_since'].clip(lower=0)
            df['recency_weight'] = np.exp(-np.log(2) * df['days_since'] / half_life)
        else:
            df['recency_weight'] = 1.0

        source_weights = self.config.get('SOURCE_WEIGHTS', {})
        df['source'] = df.get('source', '').fillna('unknown')
        df['source_weight'] = df['source'].map(source_weights).fillna(1.0)

        df['interaction_value_raw'] = np.log1p(df['price'] * df['quantity']).replace([np.inf, -np.inf], 0)
        df['interaction_value_raw'] = df['interaction_value_raw'].fillna(0)

        df['rating'] = df['interaction_value_raw'] * df['recency_weight'] * df['source_weight']
        df = df[df['rating'] > 0]

        if df.empty:
            self.logger.warning("交互数据计算后为空，可能缺少价格或数量信息")
            return pd.DataFrame(columns=["user_id", "item_id", "rating"])

        aggregated = (
            df.groupby(['user_id', 'item_id'])
              .agg(
                  rating=('rating', 'sum'),
                  interaction_strength=('interaction_value_raw', 'sum'),
                  total_spend=('price', 'sum'),
                  total_quantity=('quantity', 'sum'),
                  last_event_time=('event_time', 'max'),
                  sources=('source', lambda x: ';'.join(sorted(set(filter(None, x)))))
              )
              .reset_index()
        )

        aggregated['timestamp'] = aggregated['last_event_time']
        self.logger.info(
            "交互数据预处理完成: %s 条记录，涉及 %s 个用户和 %s 个数据集",
            len(aggregated),
            aggregated['user_id'].nunique(),
            aggregated['item_id'].nunique()
        )
        return aggregated

    def preprocess_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """兼容旧接口，直接委托给 preprocess_interactions"""
        return self.preprocess_interactions(df)

    # ------------------------------------------------------------------
    # 数据集特征处理
    # ------------------------------------------------------------------
    def preprocess_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据集元数据"""
        if df is None or df.empty:
            self.logger.warning("数据集信息为空")
            return pd.DataFrame()

        df = df.copy()

        for col in ['item_id', 'id']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'item_id' not in df.columns and 'id' in df.columns:
            df['item_id'] = df['id']

        text_columns = ['title', 'intro', 'description', 'tag']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
            else:
                df[col] = ''

        df['tag_tokens'] = df['tag'].str.replace(';', ' ', regex=False)
        df['title_clean'] = df['title'].apply(self._clean_text)
        df['description_clean'] = df['description'].apply(self._clean_text)
        df['intro_clean'] = df['intro'].apply(self._clean_text)
        df['tag_clean'] = df['tag_tokens'].apply(self._clean_text)
        df['combined_text'] = (
            df['title'] + ' ' + df['intro'] + ' ' + df['description'] + ' ' + df['tag_tokens']
        ).str.strip()
        df['combined_text_clean'] = df['combined_text'].apply(self._clean_text)

        numeric_columns = ['price', 'original_price', 'sales_volume', 'clout', 'dataset_size', 'record_count']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
            else:
                df[col] = 0
                df[f'{col}_log'] = 0

        categorical_columns = ['type_name', 'data_format', 'file_pattern', 'create_company_name']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
            else:
                df[col] = 'unknown'

        df['tag_list'] = df['tag'].apply(lambda x: [t.strip() for t in x.split(';') if t.strip()])

        self.logger.info("数据集特征预处理完成: %s 条记录", len(df))
        return df

    def create_item_features_matrix(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """创建数据集特征向量矩阵"""
        if items_df is None or items_df.empty:
            return pd.DataFrame()

        features: List[np.ndarray] = []

        numeric_cols = [
            'price', 'original_price', 'sales_volume', 'clout', 'dataset_size', 'record_count',
            'price_log', 'original_price_log', 'sales_volume_log', 'clout_log', 'dataset_size_log', 'record_count_log'
        ]
        for feature in numeric_cols:
            if feature in items_df.columns:
                scaler = StandardScaler()
                values = items_df[[feature]].fillna(0)
                scaled = scaler.fit_transform(values)
                features.append(scaled)
                self.scalers[f'item_{feature}'] = scaler

        categorical_features = ['type_name', 'data_format', 'file_pattern', 'create_company_name']
        for feature in categorical_features:
            if feature in items_df.columns:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(items_df[feature])
                one_hot = pd.get_dummies(encoded, prefix=feature)
                features.append(one_hot.values)
                self.encoders[f'item_{feature}'] = encoder

        if 'tag' in items_df.columns:
            tag_dummies = items_df['tag'].str.get_dummies(sep=';')
            if not tag_dummies.empty:
                features.append(tag_dummies.values)

        if 'combined_text_clean' in items_df.columns:
            vectorizer = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(2, 4))
            text_features = vectorizer.fit_transform(items_df['combined_text_clean'])
            features.append(text_features.toarray())
            self.vectorizers['item_text'] = vectorizer

        if not features:
            return pd.DataFrame(index=items_df['item_id'])

        feature_matrix = np.hstack(features)
        return pd.DataFrame(feature_matrix, index=items_df['item_id'])

    # ------------------------------------------------------------------
    # 用户特征处理
    # ------------------------------------------------------------------
    def preprocess_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理用户画像数据"""
        if df is None or df.empty:
            self.logger.warning("用户信息为空")
            return pd.DataFrame()

        df = df.copy()
        df['user_id'] = pd.to_numeric(df.get('user_id'), errors='coerce')
        df = df.dropna(subset=['user_id'])

        df['sex'] = df.get('sex').map({1: 'male', 2: 'female', 0: 'unknown'}).fillna('unknown')
        df['province'] = df.get('province', '').fillna('unknown').replace('', 'unknown')
        df['city'] = df.get('city', '').fillna('unknown').replace('', 'unknown')
        df['country'] = df.get('country', '').fillna('unknown').replace('', 'unknown')
        df['company_industry'] = df.get('company_industry', '').fillna('unknown').replace('', 'unknown')
        df['company_name'] = df.get('company_name', '').fillna('unknown').replace('', 'unknown')

        df['reg_channel'] = pd.to_numeric(df.get('reg_channel'), errors='coerce').fillna(0)
        df['is_consumption'] = pd.to_numeric(df.get('is_consumption'), errors='coerce').fillna(0)
        df['is_certificated'] = pd.to_numeric(df.get('is_certificated'), errors='coerce').fillna(0)

        df['location'] = (df['province'] + ' ' + df['city']).str.strip()
        df['location'] = df['location'].replace('', 'unknown')

        df['profile_text'] = (
            df['company_industry'] + ' ' + df['company_name'] + ' ' + df['location']
        ).str.strip()
        df['profile_text_clean'] = df['profile_text'].apply(self._clean_text)

        self.logger.info("用户特征预处理完成: %s 条记录", len(df))
        return df

    def create_user_features_matrix(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """创建用户特征向量矩阵"""
        if users_df is None or users_df.empty:
            return pd.DataFrame()

        features: List[np.ndarray] = []

        numeric_features = ['reg_channel', 'is_consumption', 'is_certificated']
        for feature in numeric_features:
            if feature in users_df.columns:
                scaler = StandardScaler()
                values = users_df[[feature]].fillna(0)
                scaled = scaler.fit_transform(values)
                features.append(scaled)
                self.scalers[f'user_{feature}'] = scaler

        categorical_features = ['sex', 'province', 'city', 'country']
        for feature in categorical_features:
            if feature in users_df.columns:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(users_df[feature])
                one_hot = pd.get_dummies(encoded, prefix=feature)
                features.append(one_hot.values)
                self.encoders[f'user_{feature}'] = encoder

        if 'company_industry' in users_df.columns:
            industry_dummies = users_df['company_industry'].str.get_dummies(sep=';')
            if not industry_dummies.empty:
                features.append(industry_dummies.values)

        if 'profile_text_clean' in users_df.columns:
            vectorizer = TfidfVectorizer(max_features=3000, analyzer='char_wb', ngram_range=(2, 4))
            text_features = vectorizer.fit_transform(users_df['profile_text_clean'])
            features.append(text_features.toarray())
            self.vectorizers['user_profile'] = vectorizer

        if not features:
            return pd.DataFrame(index=users_df['user_id'])

        feature_matrix = np.hstack(features)
        return pd.DataFrame(feature_matrix, index=users_df['user_id'])

    # ------------------------------------------------------------------
    # 统计与辅助方法
    # ------------------------------------------------------------------
    def create_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建用户-数据集交互矩阵"""
        if df is None or df.empty:
            return pd.DataFrame()

        matrix = df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        return matrix

    def calculate_item_similarity(self, item_features: pd.DataFrame) -> pd.DataFrame:
        """计算物品相似度矩阵"""
        similarity_matrix = cosine_similarity(item_features.values)
        return pd.DataFrame(similarity_matrix, index=item_features.index, columns=item_features.index)

    def calculate_user_similarity(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """计算用户相似度矩阵"""
        similarity_matrix = cosine_similarity(user_features.values)
        return pd.DataFrame(similarity_matrix, index=user_features.index, columns=user_features.index)

    def create_interaction_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """基于交互数据的衍生特征"""
        if ratings_df is None or ratings_df.empty:
            return pd.DataFrame()

        features: List[pd.DataFrame] = []

        user_stats = ratings_df.groupby('user_id').agg({
            'rating': ['count', 'sum'],
            'total_spend': 'sum',
            'total_quantity': 'sum'
        }).round(3)
        user_stats.columns = ['user_interaction_count', 'user_total_rating', 'user_total_spend', 'user_total_quantity']
        features.append(user_stats.fillna(0))

        item_stats = ratings_df.groupby('item_id').agg({
            'rating': ['count', 'sum'],
            'total_spend': 'sum',
            'total_quantity': 'sum'
        }).round(3)
        item_stats.columns = ['item_interaction_count', 'item_total_rating', 'item_total_revenue', 'item_total_quantity']
        features.append(item_stats.fillna(0))

        time_column = 'last_event_time' if 'last_event_time' in ratings_df.columns else None
        if time_column:
            time_features = ratings_df.groupby('user_id')[time_column].max().to_frame('last_event_time')
            time_features['days_since_last_event'] = (
                pd.Timestamp.utcnow() - time_features['last_event_time']
            ).dt.total_seconds() / 86400
            time_features['days_since_last_event'] = time_features['days_since_last_event'].clip(lower=0).fillna(0)
            features.append(time_features.drop(columns=['last_event_time']))

        return pd.concat(features, axis=1).fillna(0)

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割训练与验证数据"""
        if df is None or df.empty:
            return df, pd.DataFrame()

        from sklearn.model_selection import train_test_split

        stratify_col = df['rating'] if 'rating' in df.columns and df['rating'].nunique() > 1 else None
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        self.logger.info("数据分割完成: 训练集 %s 条，测试集 %s 条", len(train_df), len(test_df))
        return train_df, test_df

    def _clean_text(self, text: Any) -> str:
        """清理文本，保留中英文与数字"""
        if pd.isna(text):
            return ''
        text = str(text).lower()
        text = re.sub(r'[^\w\s一-鿿]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据摘要信息"""
        if df is None or df.empty:
            return {}

        summary: Dict[str, Any] = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }

        if 'rating' in df.columns:
            summary['rating_stats'] = {
                'min': df['rating'].min(),
                'max': df['rating'].max(),
                'mean': df['rating'].mean(),
                'std': df['rating'].std()
            }

        if 'user_id' in df.columns:
            summary['unique_users'] = df['user_id'].nunique()
        if 'item_id' in df.columns:
            summary['unique_items'] = df['item_id'].nunique()

        return summary
