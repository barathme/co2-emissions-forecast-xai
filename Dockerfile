FROM python:3.11-slim

LABEL maintainer="barathme@yahoo.co.in"
LABEL description="Reproducibility environment for: Explainable Machine Learning for Five-Year National CO2 Emissions Forecasting from Socioeconomic Indicators"

# System dependencies required by lightgbm, catboost, and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the full pipeline. Override with `docker run ... <command>` to run
# individual scripts (e.g. `python src/models/train.py`).
CMD ["python", "scripts/run_full_pipeline.py"]
