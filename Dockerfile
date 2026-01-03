FROM python:3.10-slim

# Install system dependencies
# ffmpeg is required for video processing
# git is useful for some pip installs if needed
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for generated content and ensure they're writable
RUN mkdir -p course_material_service/generated_videos && \
    mkdir -p course_material_service/exports && \
    mkdir -p course_material_service/data && \
    mkdir -p course_material_service/uploads && \
    chmod -R 777 course_material_service/generated_videos course_material_service/exports course_material_service/data course_material_service/uploads

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "course_material_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
