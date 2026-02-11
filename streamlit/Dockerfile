FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY index.py /app/index.py

EXPOSE 8501
CMD ["streamlit", "run", "index.py", "--server.address=0.0.0.0", "--server.port=8501"]