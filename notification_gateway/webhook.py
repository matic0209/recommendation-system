#!/usr/bin/env python3
"""
Notification Gateway for Alertmanager
接收 Alertmanager 的告警通知并转发到企业微信
"""

import json
import logging
import os
from datetime import datetime

import requests
from flask import Flask, jsonify, request

from sentry_helper import (
    capture_exception_with_context,
    capture_message_with_context,
    init_sentry,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 初始化 Sentry
sentry_enabled = init_sentry(
    service_name="notification-gateway",
    traces_sample_rate=0.5,
    profiles_sample_rate=0.5,
)
if sentry_enabled:
    logger.info("Sentry监控已启用")

# 从环境变量读取企业微信配置
CORP_ID = os.getenv('WEIXIN_CORP_ID', '')
CORP_SECRET = os.getenv('WEIXIN_CORP_SECRET', '')
AGENT_ID = int(os.getenv('WEIXIN_AGENT_ID', '1000019'))
DEFAULT_USER = os.getenv('WEIXIN_DEFAULT_USER', 'ZhangJinBo')

# 检查配置
if not CORP_ID or not CORP_SECRET:
    logger.warning("企业微信配置未设置，请检查环境变量 WEIXIN_CORP_ID 和 WEIXIN_CORP_SECRET")


def get_access_token():
    """获取企业微信 access_token"""
    url = f'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={CORP_ID}&corpsecret={CORP_SECRET}'
    try:
        # 禁用代理，直接访问企业微信 API
        response = requests.get(url, timeout=10, proxies={'http': None, 'https': None})
        result = response.json()
        if result.get('errcode') == 0:
            logger.info("成功获取企业微信 access_token")
            return result.get('access_token')
        else:
            logger.error(f"获取 access_token 失败: {result}")

            # Sentry: 记录获取 token 失败
            if sentry_enabled:
                capture_message_with_context(
                    f"企业微信 token 获取失败: errcode={result.get('errcode')}",
                    level="error",
                    error_code=result.get('errcode'),
                    error_message=result.get('errmsg'),
                )

            return None
    except Exception as e:
        logger.error(f"获取 access_token 异常: {e}")

        # Sentry: 记录获取 token 异常
        if sentry_enabled:
            capture_exception_with_context(
                e,
                level="error",
                fingerprint=["weixin", "get_access_token_failed"],
                corp_id=CORP_ID[:8] + "***",  # 脱敏
            )

        return None


def send_weixin_message(user_id, message):
    """发送企业微信消息"""
    access_token = get_access_token()
    if not access_token:
        return False

    url = f'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}'

    data = {
        "touser": user_id,
        "msgtype": "text",
        "agentid": AGENT_ID,
        "text": {
            "content": message
        },
        "safe": 0
    }

    try:
        # 禁用代理，直接访问企业微信 API
        response = requests.post(url, json=data, timeout=10, proxies={'http': None, 'https': None})
        result = response.json()
        if result.get('errcode') == 0:
            logger.info(f"消息发送成功到用户 {user_id}")
            return True
        else:
            logger.error(f"消息发送失败: {result}")

            # Sentry: 记录消息发送失败
            if sentry_enabled:
                capture_message_with_context(
                    f"企业微信消息发送失败: errcode={result.get('errcode')}",
                    level="warning",
                    user_id=user_id,
                    error_code=result.get('errcode'),
                    error_message=result.get('errmsg'),
                    message_preview=message[:100] if len(message) > 100 else message,
                )

            return False
    except Exception as e:
        logger.error(f"消息发送异常: {e}")

        # Sentry: 记录消息发送异常
        if sentry_enabled:
            capture_exception_with_context(
                e,
                level="error",
                fingerprint=["weixin", "send_message_failed"],
                user_id=user_id,
                message_preview=message[:100] if len(message) > 100 else message,
            )

        return False


def format_alert_message(alerts, receiver_name):
    """格式化告警消息"""
    messages = []

    for alert in alerts:
        status = alert.get('status', 'unknown').upper()
        labels = alert.get('labels', {})
        annotations = alert.get('annotations', {})

        # 提取关键信息
        alertname = labels.get('alertname', 'Unknown')
        severity = labels.get('severity', 'unknown').upper()
        instance = labels.get('instance', 'N/A')
        job = labels.get('job', 'N/A')
        table = labels.get('table', '')
        endpoint = labels.get('endpoint', '')

        summary = annotations.get('summary', '')
        description = annotations.get('description', '')

        # 格式化时间
        starts_at = alert.get('startsAt', '')
        if starts_at:
            try:
                dt = datetime.fromisoformat(starts_at.replace('Z', '+00:00'))
                starts_at = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass

        # 根据状态选择图标
        status_icon = '🔥' if status == 'FIRING' else '✅'
        severity_icon = {
            'CRITICAL': '🚨',
            'WARNING': '⚠️',
            'INFO': 'ℹ️'
        }.get(severity, '📢')

        # 构建消息
        msg = f"""{status_icon} [{status}] {alertname}
━━━━━━━━━━━━━━━━
{severity_icon} 严重程度: {severity}"""

        if instance != 'N/A':
            msg += f"\n🖥️  实例: {instance}"
        if job != 'N/A':
            msg += f"\n📦 任务: {job}"
        if table:
            msg += f"\n📊 表: {table}"
        if endpoint:
            msg += f"\n🔗 端点: {endpoint}"

        msg += f"\n⏰ 时间: {starts_at}"

        if summary:
            msg += f"\n\n📝 {summary}"
        if description:
            msg += f"\n{description}"

        msg += "\n━━━━━━━━━━━━━━━━"

        messages.append(msg)

    header = f"【推荐系统告警】接收器: {receiver_name}\n\n"
    return header + '\n\n'.join(messages)


@app.route('/webhook/<receiver_name>', methods=['POST'])
def webhook(receiver_name):
    """接收 Alertmanager webhook"""
    try:
        data = request.get_json()
        logger.info(f"收到 {receiver_name} 告警: {json.dumps(data, ensure_ascii=False)}")

        # 提取告警列表
        alerts = data.get('alerts', [])
        if not alerts:
            logger.info("没有告警信息")
            return jsonify({'status': 'success', 'message': 'no alerts'}), 200

        # 格式化消息
        message = format_alert_message(alerts, receiver_name)

        # 发送到企业微信
        user_id = DEFAULT_USER
        success = send_weixin_message(user_id, message)

        if success:
            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'failed to send message'}), 500

    except Exception as e:
        logger.error(f"处理 webhook 异常: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


def format_sentry_message(payload):
    """格式化 Sentry webhook 消息"""
    action = payload.get('action', 'unknown')
    data = payload.get('data', {})

    # 获取 issue 或 event 信息
    issue = data.get('issue')
    event = data.get('event')

    if not issue and not event:
        return "Sentry 告警：未知事件"

    # 提取关键信息
    title = (issue or event).get('title', 'Unknown Error')
    culprit = (issue or event).get('culprit', 'N/A')
    level = (issue or event).get('level', 'error').upper()

    # 环境和服务信息
    tags = (event or {}).get('tags', [])
    environment = next((t[1] for t in tags if t[0] == 'environment'), 'unknown')
    server_name = next((t[1] for t in tags if t[0] == 'server_name'), 'N/A')

    # Issue 统计信息
    if issue:
        count = issue.get('count', 0)
        user_count = issue.get('userCount', 0)
        first_seen = issue.get('firstSeen', '')
        last_seen = issue.get('lastSeen', '')
        permalink = issue.get('permalink', '')
        status = issue.get('status', 'unresolved')
    else:
        count = 1
        user_count = 0
        first_seen = event.get('datetime', '')
        last_seen = first_seen
        permalink = event.get('web_url', '')
        status = 'new'

    # 格式化时间
    try:
        if first_seen:
            dt = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
            first_seen_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            first_seen_str = 'N/A'
    except:
        first_seen_str = first_seen[:19] if first_seen else 'N/A'

    # 选择图标
    level_icon = {
        'FATAL': '💀',
        'ERROR': '🔥',
        'WARNING': '⚠️',
        'INFO': 'ℹ️',
        'DEBUG': '🐛'
    }.get(level, '📢')

    action_icon = {
        'created': '🆕',
        'resolved': '✅',
        'assigned': '👤',
        'ignored': '🙈'
    }.get(action.split('.')[-1], '📢')

    status_text = {
        'unresolved': '🔴 未解决',
        'resolved': '✅ 已解决',
        'ignored': '🙈 已忽略'
    }.get(status, status)

    # 构建消息
    msg = f"""{action_icon} [Sentry 应用告警] {title}
━━━━━━━━━━━━━━━━
{level_icon} 级别: {level}
🏷️  来源: Sentry
📍 环境: {environment}"""

    if server_name != 'N/A':
        msg += f"\n🖥️  服务: {server_name}"

    if culprit != 'N/A':
        msg += f"\n📝 位置: {culprit}"

    msg += f"\n⏰ 首次发现: {first_seen_str}"
    msg += f"\n📊 状态: {status_text}"

    if count > 1:
        msg += f"\n🔢 发生次数: {count}"

    if user_count > 0:
        msg += f"\n👥 影响用户: {user_count}"

    if permalink:
        msg += f"\n\n🔗 详情: {permalink}"

    msg += "\n━━━━━━━━━━━━━━━━"

    return msg


@app.route('/webhook/sentry', methods=['POST'])
def sentry_webhook():
    """接收 Sentry webhook"""
    try:
        data = request.get_json()
        logger.info(f"收到 Sentry 告警: {json.dumps(data, ensure_ascii=False)}")

        # 格式化消息
        message = format_sentry_message(data)

        # 发送到企业微信
        user_id = DEFAULT_USER
        success = send_weixin_message(user_id, message)

        if success:
            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'failed to send message'}), 500

    except Exception as e:
        logger.error(f"处理 Sentry webhook 异常: {e}", exc_info=True)

        # Sentry: 记录处理失败
        if sentry_enabled:
            capture_exception_with_context(
                e,
                level="error",
                fingerprint=["notification", "sentry_webhook_failed"],
                webhook_source="sentry",
            )

        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    config_ok = bool(CORP_ID and CORP_SECRET)
    return jsonify({
        'status': 'healthy' if config_ok else 'degraded',
        'config': {
            'corp_id': bool(CORP_ID),
            'corp_secret': bool(CORP_SECRET),
            'agent_id': AGENT_ID,
            'default_user': DEFAULT_USER
        }
    }), 200


@app.route('/test', methods=['GET', 'POST'])
def test():
    """测试发送消息"""
    message = f"""🧪 推荐系统告警测试
━━━━━━━━━━━━━━━━
⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📋 服务: Notification Gateway
✅ 状态: 测试消息
━━━━━━━━━━━━━━━━"""

    success = send_weixin_message(DEFAULT_USER, message)

    if success:
        return jsonify({'status': 'success', 'message': '测试消息已发送'}), 200
    else:
        return jsonify({'status': 'error', 'message': '测试消息发送失败'}), 500


if __name__ == '__main__':
    logger.info("启动 Notification Gateway 服务...")
    logger.info(f"企业微信配置状态: CORP_ID={'已配置' if CORP_ID else '未配置'}, CORP_SECRET={'已配置' if CORP_SECRET else '未配置'}")
    logger.info(f"默认接收人: {DEFAULT_USER}")
    app.run(host='0.0.0.0', port=9000, debug=False)
