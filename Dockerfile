# Build CPU (default):
# docker build -t tumor-api:cpu .
# docker run --rm -p 8000:8000 -v $PWD/Checkpoints:/app/Checkpoints tumor-api:cpu
# Build GPU:
# docker build --build-arg BASE_STAGE=base_gpu -t tumor-api:gpu .
# docker run --gpus all --rm -p 8000:8000 -v $PWD/Checkpoints:/app/Checkpoints tumor-api:gpu
# Change workers/port at run-time (no rebuild):
# docker run -e UVICORN_WORKERS=4 -e PORT=8080 -p 8080:8080 tumor-api:cpu
# -------------------------------------------------


# syntax=docker/dockerfile:1

# Stage 1: CPU build ##########################

# --- choose base at build time: python:*-slim by default;
# or PyTorch CUDA runtime: pytorch/pytorch:*cuda*-runtime and run with --gpus all.

ARG PYTHON_VERSION=3.11
ARG BASE_STAGE=base_cpu

# === CPU VERSION BASE ===
FROM python:${PYTHON_VERSION}-slim AS base_cpu
# === GPU VERSION BASE ===
# FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS gpu

LABEL maintainer="Somdip Sen" \
      stage="base-cpu" \
      version="1.0.0" \
      description="FastAPI brain tumor classifier for CPU"

# SHARED SYSTEM DEPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# CPU ONLY STEP -----------
# Use virtualenv to isolate deps from system python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1


# APP SETUP -----------
WORKDIR /app
COPY requirements.txt ./

# INSTALLATION -------------
# CPU: force torch CPU wheels from PyTorch repo
# GPU: default install from requirements.txt assumes CUDA
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 torchvision==0.21.0 && \
    pip install -r requirements.txt

# GPU -->
# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt

# Stage 2: GPU (CUDA) build ##########################
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS base_gpu

LABEL maintainer="Somdip Sen" \
      stage="base-gpu" \
      version="1.0.0" \
      description="FastAPI brain tumor classifier for GPU"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates curl tini && \
    rm -rf /var/lib/apt/lists/*

# Create a venv even on the GPU image so both builds look the same
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1


WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt --no-deps


# FINAL STAGE: Use whichever base was chosen
# Pick the chosen base (CPU default, or GPU via --build-arg BASE_STAGE=base_gpu)
FROM ${BASE_STAGE} AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UVICORN_WORKERS=2 \
    PORT=8000 \
    MODEL_DIR=/app/Checkpoints

# Optionally copy anything again if needed, though it's already done
# WORKDIR /app  # already set in earlier stage
# COPY --from=base_cpu /app /app  # only if needed

# Add app user (non-root)
RUN useradd -m -u 10001 appuser
USER appuser
# Copy everything now
RUN mkdir -p /app/Checkpoints && chown -R 10001:10001 /app
COPY --chown=appuser:appuser . /app

# Add health check (shared)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

EXPOSE ${PORT}
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS} --proxy-headers"]

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
