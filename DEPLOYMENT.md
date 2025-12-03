# Deployment Guide for Google Cloud

This guide explains how to deploy the Course Generator application to Google Cloud Run.

## Prerequisites

1.  **Google Cloud Project**: You need a Google Cloud project with billing enabled.
2.  **Google Cloud SDK**: Install the `gcloud` CLI tool.
3.  **Docker**: (Optional) If you want to build locally.

## 1. Enable Required APIs

Run the following commands to enable the necessary Google Cloud services:

```bash
gcloud services enable run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    sqladmin.googleapis.com \
    secretmanager.googleapis.com
```

## 2. Set Up Database (Cloud SQL)

The application requires a PostgreSQL database.

1.  **Create a Cloud SQL Instance**:
    ```bash
    gcloud sql instances create course-db-instance \
        --database-version=POSTGRES_15 \
        --cpu=1 \
        --memory=4GiB \
        --region=us-central1
    ```

2.  **Create a Database**:
    ```bash
    gcloud sql databases create course_db --instance=course-db-instance
    ```

3.  **Create a User**:
    ```bash
    gcloud sql users create course_user \
        --instance=course-db-instance \
        --password=YOUR_SECURE_PASSWORD
    ```

## 3. Set Up Vector Database (Qdrant)

For the RAG (Retrieval-Augmented Generation) features, you need a Qdrant instance.
- **Option A**: Use [Qdrant Cloud](https://cloud.qdrant.io/) (Recommended). Get the Cluster URL and API Key.
- **Option B**: Deploy Qdrant on Cloud Run (Advanced).

## 4. Deploy to Cloud Run

You can deploy directly from the source code using Cloud Build.

### Step 4.1: Submit Build

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/course-generator
```

*(Replace `YOUR_PROJECT_ID` with your actual project ID)*

### Step 4.2: Deploy Service

Deploy the container to Cloud Run, setting the necessary environment variables.

```bash
gcloud run deploy course-generator \
    --image gcr.io/YOUR_PROJECT_ID/course-generator \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars="DATABASE_URL=postgresql+asyncpg://course_user:YOUR_SECURE_PASSWORD@/course_db?host=/cloudsql/YOUR_PROJECT_ID:us-central1:course-db-instance" \
    --set-env-vars="OPENAI_API_KEY=sk-..." \
    --set-env-vars="QDRANT_URL=https://your-qdrant-cluster-url" \
    --set-env-vars="QDRANT_API_KEY=your-qdrant-key" \
    --add-cloudsql-instances="YOUR_PROJECT_ID:us-central1:course-db-instance"
```

**Important Notes:**
- **DATABASE_URL**: When using Cloud SQL with Cloud Run, the host is a Unix socket: `/cloudsql/INSTANCE_CONNECTION_NAME`. The format is `postgresql+asyncpg://USER:PASSWORD@/DB_NAME?host=/cloudsql/INSTANCE_CONNECTION_NAME`.
- **INSTANCE_CONNECTION_NAME**: Find this in the Google Cloud Console under SQL > Overview. It looks like `project:region:instance`.

## 5. Verify Deployment

1.  Get the Service URL from the output of the deploy command.
2.  Visit the URL in your browser.
3.  The database tables will be created automatically on the first run if `init_db` is called (check `main.py` startup events).

## Troubleshooting

- **Logs**: View logs in the Google Cloud Console > Cloud Run > Logs.
- **Database Connection**: Ensure the Cloud Run service account has the "Cloud SQL Client" role.
- **Memory**: If video generation crashes, increase memory: `--memory 4Gi`.
