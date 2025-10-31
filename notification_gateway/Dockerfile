FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY webhook.py .
COPY sentry_helper.py .

# 暴露端口
EXPOSE 9000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:9000/health', timeout=2)"

# 启动服务
CMD ["gunicorn", "webhook:app", "--bind", "0.0.0.0:9000", "--workers", "2", "--timeout", "30", "--access-logfile", "-", "--error-logfile", "-"]
