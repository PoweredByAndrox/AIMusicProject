FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/upgrade_whisperx_checkpoint.py /tmp/upgrade_whisperx_checkpoint.py
RUN python /tmp/upgrade_whisperx_checkpoint.py /usr/local/lib/python3.11/site-packages/whisperx/assets/pytorch_model.bin \
    && rm -f /tmp/upgrade_whisperx_checkpoint.py
COPY app ./app
COPY scripts/prefetch_lyrics_models.py /app/scripts/prefetch_lyrics_models.py
COPY scripts/run_lyrics_task.py /app/scripts/run_lyrics_task.py
COPY scripts/start.sh /app/scripts/start.sh
RUN chmod +x /app/scripts/start.sh
CMD ["/app/scripts/start.sh"]
