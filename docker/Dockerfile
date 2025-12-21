# --------------------------------------------------
# fABBA Jupyter Dockerfile
# Location: docker/Dockerfile
# --------------------------------------------------

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip \
 && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi \
 && pip install --no-cache-dir jupyter

EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook",
            "--ip=0.0.0.0",
            "--port=8888",
            "--no-browser",
            "--allow-root"]
