# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11

# Set up non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code
COPY --chown=user . /app

# Copy logical test data (EXPECTING it to be in ./data/test relative to hf_server)
COPY --chown=user ./data/test/logical_model_test_set.json /app/data/test/logical_model_test_set.json

# Create necessary directories (including templates and data/test)
RUN mkdir -p /app/data/raw /app/data/processed /app/data/test /app/trained_models/xgboost/with_domain/basic/standard/model /app/trained_models/xgboost/with_domain/basic/standard/config /app/results/latest /app/results/archive

# Set environment variables
ENV PYTHONPATH=/app

# Make log monitor script executable
RUN chmod +x /app/simple_log_monitor.sh

# Start uvicorn server (log monitoring is handled by app.py startup event)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"] 