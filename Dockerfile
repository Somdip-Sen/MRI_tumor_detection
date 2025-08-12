# Dockerfile.cpu
# ----------------
# --- choose base at build time: python:*-slim by default;
# or PyTorch CUDA runtime: pytorch/pytorch:*cuda*-runtime and run with --gpus all.
# curl, ca-certificates, and lightweight init process(tini)
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# 1) System deps kept minimal (no compilers in final)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2) Environment (fast pip, sane runtime)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=2 \
    PORT=8000 \
    MODEL_DIR=/app/Checkpoints

# 3) Install deps in a virtualenv to keep image clean
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 4) Leverage docker layer cache: copy only reqs first
WORKDIR /app
COPY requirements.txt ./

# 5) Install: CPU wheels for torch/torchvision; rest from PyPI
# Pin torch via index-url to avoid pulling CUDA wheels accidentally.
# Forces CPU wheels; PyTorch publishes CPU wheels under that index.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# 6) Create non-root user container
RUN useradd -m -u 10001 appuser
USER appuser

# 7) Add application code (after deps to keep cache warm)
COPY --chown=appuser:appuser . /app

# 8) Healthcheck (assumes /health route exists)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

EXPOSE ${APP_PORT}
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${APP_PORT} --workers ${UVICORN_WORKERS} --proxy-headers"]




### Basic steamline, replaced by code above
## Step 1: Use an official Miniconda image as the base
#FROM continuumio/miniconda3
#
## Step 2: Set the working directory in the container
#WORKDIR /app
#
## Step 3: Copy your environment file into the container
#COPY environment.yml .
#
## Step 4: Create the Conda environment from the environment.yml file
## This command reads the file and installs all specified packages
#RUN conda env create -f environment.yml
#
## Step 5: Copy your application code and assets into the container
## This includes your main.py, Checkpoints folder, Extra_transform.py etc.
#COPY . .
#
## Step 6: Expose the port the app runs on
#EXPOSE 8000
#
## Step 7: Define the command to run your app
## It first activates the shell for conda, then runs the ad-hoc command inside your environment
#CMD ["conda", "run", "-n", "mri-env", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
