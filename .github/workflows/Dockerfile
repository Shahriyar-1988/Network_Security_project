# --- Stage 1: Build environment with dependencies ---
FROM python:3.12-slim-bookworm AS builder

WORKDIR /Docker_image

# Install system dependencies, if needed (e.g. for sklearn/imaging)
RUN apt-get update && apt-get install -y build-essential 

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . .

# --- Stage 2: Final image ---
FROM python:3.12-slim-bookworm

WORKDIR /Docker_image

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /Docker_image /Docker_image

# Ensure Docker_image directory is in Python path
ENV PYTHONPATH=/Docker_image

# Default command runs batch predictions
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

