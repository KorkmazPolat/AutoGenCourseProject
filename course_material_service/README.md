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

1. cd .\course_material_service\ (doğru dizine gitme)
2. pip install -r requirements.txt  (gereklilikleri yükleme)
3. mkdir export    (kursların kaydolucağı folder oluşturma "course_material_service" içerisine) 
4. uvicorn main:app --reload    (çalıştırma)
  Uygulama `http://localhost:8000` adresinde çalışır.
5. (opsiyonel) split terminal den "docker run -p 6333:6333 qdrant/qdrant" ile vectör veri tabanı başlatılır.
  5. adım için "docker desktop uygulaması yüklü olmak zorundadır."

Note: video_builder.py dosyasında 8. satır "from moviepy.editor import (" da "moviepy.editor" olması sıkıntı çıkarırsa "moviepy" kullanmak çözebilir.

## RAG (Retrieval-Augmented Generation) Desteği

Bu sürümle birlikte uygulama PDF formatındaki kaynak dokümanları Qdrant vektör veritabanına aktarır ve içerik üretirken bu dokümanlardan alıntı yapar. Süreç şu adımları izler:

1. Web arayüzünde yer alan **Course Materials (PDF)** alanına sürükleyip bırakarak veya dosya seçerek PDF yükleyin.
2. Dosya arka planda OpenAI gömlemeleriyle parçalanır ve Docker üzerinde çalışan Qdrant koleksiyonuna yazılır.
3. Kurs üretim istekleri sırasında aynı koleksiyondan ilgili parçalar çekilir ve LLM istemine eklenir.

### Qdrant'i Docker ile Başlatma

split terminal den 
"docker run -p 6333:6333 qdrant/qdrant" ile vectör veri tabanı başlatılır.

Varsayılan olarak uygulama `http://localhost:6333` adresine bağlanır. Farklı bir barındırıcı/port kullanıyorsanız aşağıdaki ortam değişkenlerini ayarlayabilirsiniz.

### Yeni Ortam Değişkenleri

- `QDRANT_URL` veya `QDRANT_HOST`/`QDRANT_PORT`: Qdrant bağlantısı
- `QDRANT_API_KEY`: Yetkilendirme gerekiyorsa anahtar
- `QDRANT_COLLECTION` (varsayılan `course_material_docs`)
- `QDRANT_EMBED_MODEL` (varsayılan `text-embedding-3-small`)
- `QDRANT_CHUNK_SIZE` ve `QDRANT_CHUNK_OVERLAP`: PDF parçalama ayarları
- `QDRANT_TOP_K`, `QDRANT_MAX_CHARS`, `QDRANT_MIN_SCORE`: sorgu sırasında kullanılacak parametreler
- `RAG_ENABLED=true|false`: RAG'i tümden aç/kapat
- `RAG_KEEP_UPLOADS=true|false`: Yüklenen PDF dosyalarını `course_material_service/uploads/` klasöründe sakla veya ingest işleminden sonra sil

### Web Arayüzü

Formu doldururken PDF yükleme alanı belgeyi otomatik olarak Qdrant'a aktarır ve durum mesajını ekrandan takip edebilirsiniz. Kurs üretimi tamamlandığında kullanılan kaynak pasajları sonuç sayfasında **Source Context** bölümünde görülür.

Yanıt örneği:

```json
{
  "status": "indexed",
  "filename": "document.pdf",
  "pages": 18,
  "chunks": 42,
  "collection": "course_material_docs"
}
```

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
2. **Course Materials (PDF)** alanına kaynak dokümanınızı (PDF) sürükleyip bırakın veya dosya seçin. Sistem yükleme durumunu gösterecektir.
3. Opsiyonel olarak ses (örn. `alloy`) ve TTS modeli (`gpt-4o-mini-tts`) seçin.
4. İlgili aksiyon düğmesine tıklayın (plan, modül sayfası, video, tam kurs).

Arka planda `video_script` promptu çalıştırılır, slayt görselleri ve seslendirme üretildikten sonra video oynatıcıya gömülür. Dosya aynı zamanda `/videos/<dosya_adı>` yolundan indirilebilir durumdadır.

## Gereksinimler

- `ffmpeg` sisteminizde kurulu olmalı (MoviePy video render işlemleri için).
- Qdrant 1.7+ (Docker veya bulut), inbound bağlantı için 6333 portu
- OpenAI gömme ve büyük dil modellerine erişim
- PDF işlemek için `pypdf`, çok parçalı form yüklemeleri için `python-multipart`, vektör veritabanı bağlantısı için `qdrant-client` (requirements dosyasında yer alır)
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
