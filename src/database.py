"""
数据库管理模块
"""
import logging
from typing import Optional, Dict, Any, List

import pandas as pd
import pymysql  # noqa: F401 保留以确保依赖可用
from sqlalchemy import create_engine, text


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, config):
        self.config = config
        self.engine = None
        self.connection = None
        self.logger = logging.getLogger(__name__)
        self._connect()

    def _connect(self):
        """建立数据库连接"""
        try:
            connection_string = (
                f"mysql+pymysql://{self.config['DB_USER']}:{self.config['DB_PASSWORD']}"
                f"@{self.config['DB_HOST']}:{self.config['DB_PORT']}/{self.config['DB_NAME']}"
            )
            self.engine = create_engine(connection_string, echo=False)
            self.connection = self.engine.connect()
            self.logger.info("数据库连接成功")
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """执行查询并返回 DataFrame"""
        try:
            if params:
                result = pd.read_sql(text(query), self.connection, params=params)
            else:
                result = pd.read_sql(text(query), self.connection)
            return result
        except Exception as e:
            self.logger.error(f"查询执行失败: {str(e)}")
            raise

    def _interaction_subquery(self) -> str:
        """构建用户与数据集交互的聚合查询"""
        return """
            SELECT
                o.create_user        AS user_id,
                o.dataset_id         AS item_id,
                o.price              AS price,
                1                    AS quantity,
                o.pay_status         AS pay_status,
                o.task_status        AS task_status,
                o.create_time        AS create_time,
                COALESCE(o.pay_time, o.update_time, o.create_time) AS event_time,
                'dataset_order'      AS source
            FROM order_tab o
            WHERE o.is_delete = 0
              AND o.dataset_id IS NOT NULL
              AND o.create_user IS NOT NULL
              AND o.pay_status IN (1,5)
        UNION ALL
            SELECT
                a.creator_id         AS user_id,
                a.api_id             AS item_id,
                a.price              AS price,
                COALESCE(a.frequency, 1) AS quantity,
                a.pay_status         AS pay_status,
                a.run_status         AS task_status,
                a.create_time        AS create_time,
                COALESCE(a.pay_time, a.update_time, a.create_time) AS event_time,
                'api_order'          AS source
            FROM api_order a
            WHERE a.is_delete = 0
              AND a.api_id IS NOT NULL
              AND a.creator_id IS NOT NULL
              AND a.pay_status = 1
        """

    # ------------------------------------------------------------------
    # 交互数据
    # ------------------------------------------------------------------
    def get_user_ratings(self, user_id: Optional[int] = None) -> pd.DataFrame:
        """获取用户与数据集的历史交互记录"""
        base_query = f"""
        SELECT *
        FROM (
            {self._interaction_subquery()}
        ) interactions
        WHERE interactions.user_id IS NOT NULL
        """

        params = None
        if user_id is not None:
            base_query += " AND interactions.user_id = :user_id"
            params = {'user_id': user_id}

        return self.execute_query(base_query, params)

    def get_interaction_data(self) -> pd.DataFrame:
        """获取用户、数据集与交互的联合数据"""
        query = f"""
        SELECT
            interactions.user_id,
            interactions.item_id,
            interactions.price,
            interactions.quantity,
            interactions.event_time,
            interactions.source,
            items.title,
            items.price AS item_price,
            items.type_name,
            users.user_name,
            users.company_name
        FROM ({self._interaction_subquery()}) interactions
        LEFT JOIN (
            SELECT
                d.id                         AS item_id,
                d.dataset_name               AS title,
                d.price                      AS price,
                dict_table.data_value        AS type_name
            FROM dataset d
            LEFT JOIN dict dict_table ON dict_table.id = d.type_id
            WHERE d.is_delete = 0
        ) items ON interactions.item_id = items.item_id
        LEFT JOIN (
            SELECT
                u.id           AS user_id,
                u.user_name    AS user_name,
                u.company_name AS company_name
            FROM user u
            WHERE u.is_valid = 1
        ) users ON interactions.user_id = users.user_id
        """
        return self.execute_query(query)

    def get_user_item_matrix(self) -> pd.DataFrame:
        """获取用户-物品交互矩阵"""
        ratings_df = self.get_user_ratings()
        if ratings_df.empty:
            return pd.DataFrame()

        ratings_df = ratings_df[['user_id', 'item_id', 'price', 'quantity']].copy()
        ratings_df['price'] = pd.to_numeric(ratings_df['price'], errors='coerce').fillna(0)
        ratings_df['quantity'] = pd.to_numeric(ratings_df['quantity'], errors='coerce').fillna(1)
        ratings_df['interaction_value'] = ratings_df['price'] * ratings_df['quantity']

        pivot_source = ratings_df.groupby(['user_id', 'item_id'], as_index=False)['interaction_value'].sum()

        return pivot_source.pivot_table(
            index='user_id',
            columns='item_id',
            values='interaction_value',
            fill_value=0
        )

    def get_popular_items(self, limit: int = 100) -> pd.DataFrame:
        """基于交互热度统计热门数据集"""
        query = f"""
        SELECT
            interactions.item_id,
            COUNT(*) AS interaction_count,
            COUNT(DISTINCT interactions.user_id) AS unique_users,
            SUM(COALESCE(interactions.price, 0) * COALESCE(interactions.quantity, 1)) AS total_revenue,
            MAX(interactions.event_time) AS last_interaction
        FROM ({self._interaction_subquery()}) interactions
        GROUP BY interactions.item_id
        ORDER BY interaction_count DESC, total_revenue DESC
        LIMIT :limit
        """
        return self.execute_query(query, {'limit': limit})

    def get_cold_start_users(self) -> List[int]:
        """获取交互次数少于5次的用户"""
        query = f"""
        SELECT interactions.user_id, COUNT(*) AS interaction_count
        FROM ({self._interaction_subquery()}) interactions
        GROUP BY interactions.user_id
        HAVING interaction_count < 5
        """
        df = self.execute_query(query)
        return df['user_id'].tolist()

    def get_cold_start_items(self) -> List[int]:
        """获取交互次数少于5次的数据集"""
        query = f"""
        SELECT interactions.item_id, COUNT(*) AS interaction_count
        FROM ({self._interaction_subquery()}) interactions
        GROUP BY interactions.item_id
        HAVING interaction_count < 5
        """
        df = self.execute_query(query)
        return df['item_id'].tolist()

    # ------------------------------------------------------------------
    # 数据集特征
    # ------------------------------------------------------------------
    def get_item_features(self, item_id: Optional[int] = None) -> pd.DataFrame:
        """获取数据集（物品）特征数据"""
        query = """
        SELECT
            d.id                             AS item_id,
            d.dataset_name                   AS title,
            d.description_txt                AS description,
            d.intro                          AS intro,
            d.tag                            AS tag,
            d.type_id                        AS type_id,
            dict_table.data_value            AS type_name,
            d.data_format                    AS data_format,
            d.pattern                        AS file_pattern,
            d.price                          AS price,
            d.original_price                 AS original_price,
            d.sales_volume                   AS sales_volume,
            d.clout                          AS clout,
            d.amount                         AS record_count,
            d.dataset_size                   AS dataset_size,
            d.create_company_name            AS create_company_name,
            d.status                         AS audit_status,
            d.publish_status                 AS publish_status,
            d.create_time                    AS create_time,
            d.update_time                    AS update_time,
            (
                SELECT di.image_url
                FROM dataset_image di
                WHERE di.dataset_id = d.id
                ORDER BY di.image_order
                LIMIT 1
            )                                AS cover_image
        FROM dataset d
        LEFT JOIN dict dict_table ON dict_table.id = d.type_id
        WHERE d.is_delete = 0
        """

        params = None
        if item_id is not None:
            query += " AND d.id = :item_id"
            params = {'item_id': item_id}

        return self.execute_query(query, params)

    # ------------------------------------------------------------------
    # 用户特征
    # ------------------------------------------------------------------
    def get_user_features(self, user_id: Optional[int] = None) -> pd.DataFrame:
        """获取用户特征数据"""
        query = """
        SELECT
            u.id                 AS user_id,
            u.user_name          AS user_name,
            u.mobile             AS mobile,
            u.user_email         AS user_email,
            u.reg_channel        AS reg_channel,
            u.role               AS role,
            u.is_valid           AS is_valid,
            u.is_certificated    AS is_certificated,
            u.is_register        AS is_register,
            u.is_consumption     AS is_consumption,
            u.company_id         AS company_id,
            u.company_name       AS company_name,
            u.province           AS province,
            u.city               AS city,
            u.country            AS country,
            u.sex                AS sex,
            u.last_login_time    AS last_login_time,
            u.create_time        AS create_time,
            c.industry           AS company_industry,
            c.address            AS company_address,
            c.website            AS company_website,
            c.status             AS company_status
        FROM user u
        LEFT JOIN company c ON u.company_id = c.id
        WHERE u.is_valid = 1
        """

        params = None
        if user_id is not None:
            query += " AND u.id = :user_id"
            params = {'user_id': user_id}

        return self.execute_query(query, params)

    # ------------------------------------------------------------------
    # 其他
    # ------------------------------------------------------------------
    def insert_rating(self, *args, **kwargs):
        """当前库表未直接存储评分信息，该接口暂未实现。"""
        raise NotImplementedError("当前数据库未提供显式评分表，无法直接写入评分。")

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        self.logger.info("数据库连接已关闭")
