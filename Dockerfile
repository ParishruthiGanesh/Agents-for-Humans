# For Fly.io, AWS App Runner, ECS, or anywhere that takes a container.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recall/ ./recall/
COPY web/ ./web/
COPY data/ ./data/

# Replay mode by default: a public demo should not hold AWS credentials.
# To run the real agents, set RECALL_OFFLINE=0 and supply AWS credentials
# through your platform's secret store — never bake them into the image.
ENV RECALL_OFFLINE=1 RECALL_DEMO=1 PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn recall.api:app --host 0.0.0.0 --port ${PORT}"]
