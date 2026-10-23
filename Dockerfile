FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

COPY pyproject.toml README.md ./
COPY app/ ./app/

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["python", "-m", "app.agents"]
