FROM python:3.9-slim

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Oil & Gas Market Optimization - AI-Driven Market and Supply Optimization for Oil & Gas Commodities"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw \
    data/processed \
    data/features \
    data/insights \
    data/chroma \
    logs \
    results/forecasting \
    results/backtests \
    results/model_selection \
    results/monte_carlo \
    results/trading

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit
EXPOSE 8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "dashboard" ]; then\n\
    exec python run_dashboard.py\n\
elif [ "$1" = "trading-dashboard" ]; then\n\
    exec python run_trading_dashboard.py\n\
elif [ "$1" = "data-pipeline" ]; then\n\
    exec python -m src.pipeline.main "$@"\n\
elif [ "$1" = "rag-pipeline" ]; then\n\
    exec python -m src.pipeline.rag_pipeline "$@"\n\
elif [ "$1" = "trading-pipeline" ]; then\n\
    exec python -m src.pipeline.trading_pipeline "$@"\n\
elif [ "$1" = "full-pipeline" ]; then\n\
    exec python run_full_pipeline.py "$@"\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["dashboard"]