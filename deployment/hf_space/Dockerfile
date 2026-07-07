# Hugging Face Space (Docker SDK). Exposes port 7860.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    PORT=7860

# ffmpeg = robust decoding of mp3/m4a/webm/opus uploads
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
