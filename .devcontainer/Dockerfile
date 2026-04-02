FROM python:3.10.13-bookworm
USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_FINBERT=false
RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && apt-get install -y google-cloud-cli && rm -rf /var/lib/apt/lists/*
COPY .devcontainer/requirements.txt /workspace/ofcourseistillloveyou/requirements.txt
COPY .devcontainer/requirements-finbert.txt /workspace/ofcourseistillloveyou/requirements-finbert.txt
WORKDIR /workspace/ofcourseistillloveyou
RUN PIP_DEFAULT_TIMEOUT=600 pip install --no-cache-dir -r requirements.txt
RUN if [ "$INSTALL_FINBERT" = "true" ]; then \
        PIP_DEFAULT_TIMEOUT=600 pip install --no-cache-dir -r requirements-finbert.txt; \
    fi
