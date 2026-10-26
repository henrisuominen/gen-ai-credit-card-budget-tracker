FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=127.0.0.1 \
    PORT=8000

COPY pyproject.toml README.md ./
COPY app/ ./app/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["adk", "web", "--host", "127.0.0.1", "--port", "8000"]
