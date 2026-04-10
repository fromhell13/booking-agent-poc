FROM python:3.11-slim

WORKDIR /app

COPY ingest/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# mcp_server package so ingest can import MenuRAG
COPY mcp_server /app/mcp_server
COPY ingest/ingest.py /app/ingest.py

ENV PYTHONPATH=/app

CMD ["python", "ingest.py"]
