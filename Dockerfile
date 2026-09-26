# syntax=docker/dockerfile:1
# Python services. One file, two builds (see docker-compose.yml):
#   api/init  CPU torch, E5 weights baked in so the embedder never downloads at runtime
#   cv        CUDA torch plus the cv extra, no E5
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGES=torch
ARG EXTRAS=
ARG BAKE_E5=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# torch first, from its own index: the default PyPI wheel is the multi-GB CUDA build.
RUN pip install --index-url "$TORCH_INDEX" $TORCH_PACKAGES

# Dependencies from pyproject.toml alone, so a source change does not reinstall them.
COPY pyproject.toml .
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; \
extras = [e for e in '$EXTRAS'.split(',') if e]; \
deps = p['dependencies'] + [d for e in extras for d in p['optional-dependencies'][e]]; \
open('/tmp/requirements.txt', 'w').write('\n'.join(deps))" \
 && pip install -r /tmp/requirements.txt

# LocalE5Embedder loads with local_files_only=True, so the weights must be in the cache.
RUN if [ "$BAKE_E5" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')"; \
    fi

COPY alembic.ini .
COPY alembic ./alembic
COPY apps ./apps
COPY scripts ./scripts
COPY evals ./evals
COPY docker ./docker
# A Windows checkout with core.autocrlf turns the script into CRLF, which sh rejects.
RUN sed -i 's/\r$//' docker/*.sh

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "apps.main_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
