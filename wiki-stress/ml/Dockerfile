FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /usr/local/bin/uv
WORKDIR /app
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.8.0"
COPY etl /app/etl
COPY ml /app/ml
# uv, not pip, resolves this. optimum 2.x serves its `onnxruntime` extra from
# the separate optimum-onnx distribution, and pip 25 cannot satisfy
# `optimum[onnxruntime]` that way: it backtracks across optimum 1.24-2.3 and
# transformers 4.45+ and then fails with ResolutionImpossible. uv resolves the
# same requirement -- it is what produced uv.lock. Torch stays pip-installed
# above so the CPU wheel is kept; uv leaves an already-satisfied requirement
# alone.
RUN uv pip install --system --no-cache /app/etl /app/ml
# Only uk_core_news_trf needs this; the sm and lg pipelines do not. It is a
# separate step so a resolver change here cannot disturb the line above.
RUN uv pip install --system --no-cache 'spacy-curated-transformers>=0.2.2,<1.0.0'
COPY --chown=10001:10001 models/v3-xenc /app/models/v3-xenc
RUN useradd --create-home --uid 10001 model
USER model
EXPOSE 8090
CMD ["uvicorn", "ukstress_ml.serving:create_app", "--factory", "--host", "0.0.0.0", "--port", "8090"]
