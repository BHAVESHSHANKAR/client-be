# Dockerfile
# ----- base image (TF works well on 3.11) -----
    FROM python:3.11-slim

    # Env tweaks
    ENV PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        PIP_DISABLE_PIP_VERSION_CHECK=1
    
    # Workdir
    WORKDIR /app
    
    # System deps (psycopg2/opencv/pillow etc.)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libpq-dev \
        curl \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy requirements first (better layer caching)
    COPY requirements.txt .
    
    # Install Python deps
    RUN pip install --upgrade pip setuptools wheel \
     && pip install -r requirements.txt
    
    # Copy the rest of your project
    COPY . .
    
    # Expose the port Render injects via $PORT
    EXPOSE 8000
    
    # Start your Flask app (server.py must expose `app`)
    CMD gunicorn server:app --bind 0.0.0.0:${PORT:-8000}
    