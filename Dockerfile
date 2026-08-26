FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV REPOLENS_BASE_URL=https://openrouter.ai/api/v1
EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health')" || exit 1

ENTRYPOINT ["repolens"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8787"]
