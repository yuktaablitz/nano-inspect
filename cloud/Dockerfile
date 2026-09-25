# Cloud escalation tier: human review console + label store. No GPU, no AI inference.
#   docker build -t nanoinspect-cloud -f cloud/Dockerfile .
#   docker run -p 9000:9000 -e NANOINSPECT_CLOUD_KEY=change-me -v nanoinspect-cloud:/data nanoinspect-cloud
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.136.3 uvicorn==0.53.0 pydantic==2.13.5
COPY cloud/ cloud/
ENV NANOINSPECT_CLOUD_DB=/data/cloud.db
RUN mkdir -p /data
EXPOSE 9000
CMD ["uvicorn", "cloud.server:app", "--host", "0.0.0.0", "--port", "9000"]
