FROM python:3.12-slim

RUN useradd --create-home --uid 10001 ukstress
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
USER ukstress
ENTRYPOINT ["ukstress"]

