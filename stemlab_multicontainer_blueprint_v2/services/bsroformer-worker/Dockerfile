FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/patch_bs_vendor.py /app/scripts/patch_bs_vendor.py
RUN python /app/scripts/patch_bs_vendor.py
COPY app ./app
COPY tests ./tests
COPY scripts/prefetch_bs_models.py /app/scripts/prefetch_bs_models.py
COPY scripts/start.sh /app/scripts/start.sh
RUN chmod +x /app/scripts/start.sh
CMD ["/app/scripts/start.sh"]
