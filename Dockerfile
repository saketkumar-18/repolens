FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Default to OpenRouter; any OpenAI-compatible endpoint works via REPOLENS_BASE_URL.
ENV REPOLENS_BASE_URL=https://openrouter.ai/api/v1

# $PORT is injected by hosts (Render, Railway, Fly). Default 8787 for local runs.
ENV PORT=8787

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8787')+'/health')" || exit 1

EXPOSE 8787

ENTRYPOINT ["repolens"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8787"]
