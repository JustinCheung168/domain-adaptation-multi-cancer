FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ---- System dependencies required by torch ----
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies ----
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY . .

# Railway provides $PORT
CMD ["sh", "-c", "gunicorn app:app --workers 1 --bind 0.0.0.0:$PORT"]
