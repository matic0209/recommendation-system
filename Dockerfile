# Multi-stage build for production-grade recommendation service
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Production image
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r -g 50000 appuser && \
    useradd -r -u 50000 -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV MPLCONFIGDIR=/tmp

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ /app/app/
COPY config/ /app/config/
COPY pipeline/ /app/pipeline/
COPY data/ /app/data/
COPY models/ /app/models/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/data/evaluation && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with multiple workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
