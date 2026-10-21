FROM python:3.11-slim

# Install system dependencies including Tesseract OCR and PostgreSQL client
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt pyproject.toml setup.py README.md ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy application code
COPY scrag/ ./scrag/
COPY scripts/ ./scripts/
COPY schema.sql ./

ENV PYTHONUNBUFFERED=1

# Default entrypoint runs the batch pipeline pulling from GCS V2 bucket
ENTRYPOINT ["python", "-m", "scrag", "run"]
CMD ["--from-gcs"]
