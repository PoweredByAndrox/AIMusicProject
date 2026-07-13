FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf/hub \
    OMP_NUM_THREADS=4

WORKDIR /work

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import faster_whisper; print(faster_whisper.__version__)"

ENTRYPOINT ["python", "/work/ml_worker_improved/tools/audit_minus_content_words.py"]
