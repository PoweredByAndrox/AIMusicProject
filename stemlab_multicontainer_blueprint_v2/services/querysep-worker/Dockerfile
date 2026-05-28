FROM python:3.11-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY tests ./tests
COPY scripts/prefetch_querysep_model.py /app/scripts/prefetch_querysep_model.py
COPY scripts/start.sh /app/scripts/start.sh
COPY vendor ./vendor
RUN chmod +x /app/scripts/start.sh
CMD ["/app/scripts/start.sh"]
