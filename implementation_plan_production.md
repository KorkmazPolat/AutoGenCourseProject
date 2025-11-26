# Production Implementation Plan

Turning the local AutoGen Course Project into a live, production-ready website requires several key infrastructure and code changes.

## 1. Infrastructure & Hosting

### A. Application Hosting
Move from `localhost` to a cloud provider.
*   **Recommendation:** **Railway** or **Render**. They are easiest for Python/FastAPI apps and support Docker.
*   **Alternative:** AWS App Runner (more scalable, more complex).

### B. Database (PostgreSQL)
Switch from SQLite (`course_platform.db`) to a production-grade PostgreSQL database.
*   **Why:** SQLite doesn't handle concurrent writes well (multiple users generating courses at once).
*   **Action:** Provision a Postgres instance (e.g., via Supabase, Neon, or the hosting provider's add-on).
*   **Migration:** Use `alembic` to manage schema changes.

### C. File Storage (Object Storage)
Stop saving videos/images to the local disk (`/static/videos`).
*   **Why:** Cloud servers have ephemeral file systems (files disappear on redeploy).
*   **Action:** Implement **AWS S3** (or Cloudflare R2 / Supabase Storage).
*   **Code Change:** Update `MediaEngine` and `VideoGeneratorAgent` to upload files to S3 and save the *URL* in the database, not the local path.

## 2. Backend Architecture (Scalability)

### A. Asynchronous Task Queue (Critical)
Video generation takes minutes. A web request typically times out after 30-60 seconds.
*   **Current State:** `await video_generator.generate()` runs in the web process.
*   **Production State:**
    1.  User clicks "Generate".
    2.  API pushes a job to a **Redis Queue**.
    3.  API returns "Job Started" immediately.
    4.  A separate **Worker Process** (Celery or ARQ) picks up the job and runs the heavy AI/Video tasks.
    5.  Frontend polls for status or uses WebSockets for updates.

### B. Environment Variables
Never commit secrets.
*   **Action:** Ensure all API keys (`OPENAI_API_KEY`, `DATABASE_URL`, `AWS_ACCESS_KEY`) are loaded via `os.getenv()` and set in the cloud provider's dashboard.

## 3. Frontend & User Experience

### A. Authentication
Ensure user accounts are secure.
*   **Current:** Custom JWT implementation.
*   **Action:** Verify password hashing (bcrypt) and JWT expiration. Consider adding email verification.

### B. Domain & SSL
*   **Action:** Buy a domain (e.g., `agentic-courses.com`).
*   **Action:** Configure DNS and enable SSL (HTTPS) provided by the host.

## 4. Roadmap Steps

### Phase 1: Cloud Essentials (Immediate)
1.  **S3 Integration:** Modify code to upload assets to S3.
2.  **PostgreSQL Support:** Ensure SQLAlchemy models work with Postgres.
3.  **Dockerization:** Create a `Dockerfile` for consistent deployment.

### Phase 2: Async Workers (Performance)
1.  **Redis Setup:** Spin up a Redis instance.
2.  **Task Queue:** Refactor `/generate-course` to offload work to a background worker.
3.  **Progress Tracking:** Update the frontend to poll the job status from the database/cache.

### Phase 3: Deployment
1.  **Deploy DB & Redis.**
2.  **Deploy Web Service.**
3.  **Deploy Worker Service.**
4.  **Connect Domain.**

## 5. Cost Estimate (MVP)
*   **Hosting (Railway/Render):** ~$5-10/mo
*   **Database (Supabase/Neon):** Free tier available
*   **Storage (S3/R2):** Pennies/mo (pay per GB)
*   **OpenAI API:** Pay per usage (variable)
