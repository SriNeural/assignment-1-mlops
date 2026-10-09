FROM python:3.11-slim

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY models ./models

RUN mkdir -p /app/logs && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    CARDIA_MODEL_PATH=/app/models/heart_risk_model.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "cardiascreen.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
