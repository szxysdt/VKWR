# API Endpoints

The server exposes an OpenAI-compatible API:

- `GET /v1/models` — List available models
- `POST /v1/completions` — Text completion (sync and streaming)
- `POST /v1/chat/completions` — Chat completion (sync and streaming)

## Example Requests

### List Models

```bash
curl http://localhost:8000/v1/models
```

### Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model_name>", "prompt": "Hello, how are you?", "max_tokens": 64, "temperature": 0.8}'
```

### Streaming Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model_name>", "prompt": "Hello", "max_tokens": 128, "stream": true}'
```

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model_name>", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

> Replace `<model_name>` with an actual model ID from `GET /v1/models`.
