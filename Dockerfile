FROM python:3.11-slim

WORKDIR /app

# Build tools + OpenJDK for Apache Tika
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake git ninja-build \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME so tika-python can find the JVM
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN pip install --no-cache-dir pipenv

# Install base dependencies
COPY Pipfile ./
RUN pip install --no-cache-dir \
    scikit-learn numpy fastapi "uvicorn[standard]" python-multipart \
    ripser neo4j tika pdfplumber pandas openpyxl pyyaml

# PyTorch CPU — from dedicated index
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# giotto-tda from source (cmake + git). Falls back gracefully if it fails.
RUN pip install --no-cache-dir giotto-tda || \
    echo "[WARNING] giotto-tda build failed — running with ripser fallback"

# KeplerMapper (corpus shape visualization) — install without deps to avoid openmp clash
RUN pip install --no-cache-dir kmapper --no-deps

# POT (Python Optimal Transport) — required by gudhi.wasserstein
RUN pip install --no-cache-dir pot

# GUDHI (drift detection) — no wheel for linux/arm64; falls back to mean-persistence diff if unavailable
RUN pip install --no-cache-dir gudhi || \
    echo "[WARNING] gudhi not available on this platform — drift uses mean-persistence fallback"

# Pre-download the Tika JAR so first-request latency is zero
RUN python -c "import tika; tika.initVM()" 2>/dev/null || true

COPY patternengine/ ./patternengine/
COPY hudex_tda.html ./

WORKDIR /app/patternengine

EXPOSE 8002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
