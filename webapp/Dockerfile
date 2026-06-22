# COPD Insight — container image for deployment (Hugging Face Spaces, Render,
# Railway, or any host that runs a Docker image).
#
# Hugging Face Spaces specifically expects the app to listen on port 7860 —
# that's why it's hardcoded in the CMD below rather than read from $PORT.
# If you deploy somewhere that assigns its own $PORT (e.g. Render), change
# the CMD to: gunicorn --bind 0.0.0.0:$PORT app:app

FROM python:3.11-slim

WORKDIR /app

# System libs Pillow/TensorFlow need for image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
