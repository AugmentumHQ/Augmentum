FROM python:3.11-slim-bookworm

WORKDIR /app

# Install scientific Python packages (NO network libraries).
# Wheels for numpy/pandas/scipy come from PyPI/files.pythonhosted.org and
# can stall mid-download on a slow link — scipy alone is ~35 MB. Without
# retries the whole build fails on a transient TCP hiccup. Mirror the
# main Dockerfile's resilience flags + add a BuildKit cache mount so a
# rebuild after a partial failure doesn't re-download from scratch.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 300 \
    numpy \
    pandas \
    scipy \
    sympy \
    matplotlib \
    python-dateutil \
    flask

# Copy executor code
COPY executor_server.py sandbox.py ./

# Create non-root user and working directory
RUN useradd -m -u 1000 executor && \
    mkdir -p /work && chown executor:executor /work

USER executor
WORKDIR /work

EXPOSE 5000

CMD ["python", "/app/executor_server.py"]
