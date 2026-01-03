#!/bin/bash
PROJECT_ID=$(gcloud config get-value project)
INSTANCE_NAME="course-db-instance"
DB_NAME="course_db"
DB_USER="course_user"
DB_PASS="f7S6rCEfGv4iNihI"
REGION="us-central1"
CONN_NAME="$PROJECT_ID:$REGION:$INSTANCE_NAME"

echo "Waiting for SQL instance to be ready..."
while true; do
  STATE=$(gcloud sql instances describe $INSTANCE_NAME --format="value(state)" 2>/dev/null)
  if [ "$STATE" == "RUNNABLE" ]; then
    echo "Instance is ready!"
    break
  fi
  echo "Current state: $STATE. Waiting..."
  sleep 30
done

echo "Creating database..."
gcloud sql databases create $DB_NAME --instance=$INSTANCE_NAME

echo "Creating user..."
gcloud sql users create $DB_USER --instance=$INSTANCE_NAME --password=$DB_PASS

echo "Deploying to Cloud Run..."
DATABASE_URL="postgresql+asyncpg://$DB_USER:$DB_PASS@/$DB_NAME?host=/cloudsql/$CONN_NAME"

gcloud run deploy course-generator \
    --region $REGION \
    --set-env-vars="DATABASE_URL=$DATABASE_URL" \
    --add-cloudsql-instances="$CONN_NAME" \
    --quiet

echo "Setup complete!"
