# Course Material Prompt Service

FastAPI tabanlı bu servis, çevrim içi kurs materyallerini otomatik üretmek için kullanılacak prompt şablonlarını üretir. Her bir materyal tipi için ayrı promptlar ve hepsini aynı anda oluşturan tek bir prompt sağlar.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Ardından servis `http://localhost:8000` adresinden ulaşılabilir.

> **Not:** Servisin OpenAI ile içerik üretebilmesi için `OPENAI_API_KEY` ortam değişkeninin ayarlı olması gerekir.
>
> ```bash
> export OPENAI_API_KEY="sk-..."
> ```

## Örnek İstek

```bash
curl -X POST http://localhost:8000/materials/video_script \
  -H "Content-Type: application/json" \
  -d '{
    "course_title": "AI Product Management Fundamentals",
    "learning_outcomes": [
      "Define the stages of an AI product lifecycle",
      "Identify ethical risks in data collection",
      "Craft a success metric for an AI feature"
    ],
    "audience": "Mid-level product managers transitioning into AI products",
    "tone": "Pragmatic and encouraging",
    "duration_minutes": 12,
    "project_duration": "6-8 hours"
  }'
```

Veya aynı isteği gövdeye `prompt_id` ekleyerek tek uç noktadan yapabilirsiniz:

```bash
curl -X POST http://localhost:8000/materials \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_id": "video_script",
    "course_title": "AI Product Management Fundamentals",
    "learning_outcomes": [
      "Define the stages of an AI product lifecycle",
      "Identify ethical risks in data collection",
      "Craft a success metric for an AI feature"
    ],
    "audience": "Mid-level product managers transitioning into AI products",
    "tone": "Pragmatic and encouraging",
    "duration_minutes": 12,
    "project_duration": "6-8 hours"
  }'
```

Örnek dönen değer:

```json
{
  "prompt_id": "video_script",
  "description": "Generates a JSON-formatted brief for a video lesson.",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "response_format": {
    "type": "object",
    "properties": {
      "hook": {"type": "string"},
      "outline": {"type": "array"},
      "narration": {"type": "array"},
      "recap": {"type": "string"},
      "call_to_action": {"type": "string"}
    },
    "required": ["hook", "outline", "narration", "recap", "call_to_action"]
  }
}
```

Tüm promptları aynı anda almak için `POST /materials/all` uç noktasını kullanabilirsiniz.

- Yalnızca prompt metnini görmek isterseniz `preview=true` sorgu parametresi ekleyin:

  ```bash
  curl -X POST "http://localhost:8000/materials/video_script?preview=true" \
    -H "Content-Type: application/json" \
    -d '{"course_title":"...", "learning_outcomes":["..."]}'
  ```

- Varsayılan model `gpt-4o-mini`’dir; istekte `model` alanı ile değiştirebilirsiniz.
