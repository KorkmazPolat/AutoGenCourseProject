# Course Material Prompt Service & Video Studio

FastAPI tabanlı bu uygulama hem kurs materyali üretmek için REST API uç noktaları hem de temel bir web arayüzü sunar. Kullanıcılar kurs bilgilerini girip otomatik olarak oluşturulan video çıktısını hem API üzerinden alabilir hem de tarayıcıda izleyebilir.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Not:** `OPENAI_API_KEY` ortam değişkenini ayarlamadan uygulama içerik üretemez.
>
> ```bash
> export OPENAI_API_KEY="sk-..."
> ```
> 2. yol:  bir .env dosyası oluşturun ve içine ekleyin OPENAI_API_KEY=sk-...

## Çalıştırma

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Uygulama `http://localhost:8000` adresinde çalışır.

## REST API Kullanımı

- `GET /prompts` — Kullanılabilir prompt listesini döndürür.
- `POST /materials/{prompt_id}` — İlgili prompt için içerik üretir.
- `POST /materials` — `{"prompt_id": "...", ...}` formatındaki gövde ile aynı işlemi yapar.
- `POST /materials/all` — Tüm prompt çıktılarının tek seferde alınmasını sağlar.

### Video Script Örneği

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
    "duration_minutes": 12
  }'
```

`video_script` isteği sırasında otomatik olarak `generated_videos/` dizininde mp4 dosyası üretilir ve yanıta `video_file` alanı eklenir. Video üretimini kapatmak için `create_video=false` sorgu parametresini veya gövde alanını kullanabilirsiniz. Farklı bir ses/model için `voice` ve `tts_model` parametrelerini ayarlayın.

## Web Arayüzü

Tarayıcıdan `http://localhost:8000/` adresine giderek formu doldurun:

1. Kurs başlığı, hedef kitle, ton ve öğrenme çıktıları (satır satır) girin.
2. Opsiyonel olarak ses (örn. `alloy`) ve TTS modeli (`gpt-4o-mini-tts`) seçin.
3. “Create Video” düğmesine tıklayın.

Arka planda `video_script` promptu çalıştırılır, slayt görselleri ve seslendirme üretildikten sonra video oynatıcıya gömülür. Dosya aynı zamanda `/videos/<dosya_adı>` yolundan indirilebilir durumdadır.

## Gereksinimler

- `ffmpeg` sisteminizde kurulu olmalı (MoviePy video render işlemleri için).
- OpenAI TTS modellerine erişim (`gpt-4o-mini-tts` vb.).

### ffmpeg Kurulumu (Örnekler)

- macOS (Homebrew): `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt update && sudo apt install -y ffmpeg`
- Fedora: `sudo dnf install -y ffmpeg`
- Windows: Statik ffmpeg build indirip `ffmpeg.exe` yolunu `PATH`'e ekleyin (PowerShell: Sistem Ortam Değişkenleri > PATH).

Kurulum doğrulama: `ffmpeg -version` çıktısı görülmeli. MoviePy, `imageio-ffmpeg` aracılığıyla sistemdeki ffmpeg’i bulur. Ağ kısıtlı ortamlarda otomatik indirme başarısız olursa sistem ffmpeg’i kurmak en sağlam çözümdür.

## Dizin Yapısı

```
course_material_service/
├── main.py                # FastAPI uygulaması (REST + Web)
├── prompts.yaml           # Prompt tanımları
├── video_builder.py       # Slayt + seslendirme üzerinden video üretimi
├── templates/             # Basit HTML şablonları
└── generated_videos/      # Çıktı videoları (çalışma sırasında oluşur)
```
