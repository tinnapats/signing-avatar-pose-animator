FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/data/huggingface
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements-cloud.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-cloud.txt
COPY export_pose_animator_sequence.py clip_database.py pose_comparison.py setup_comparison.py run_pose_animator_server.py cloud_backend.py ./
RUN python setup_comparison.py
EXPOSE 8000
CMD ["python", "cloud_backend.py"]
