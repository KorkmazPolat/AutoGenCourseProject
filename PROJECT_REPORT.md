<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/tr/6/69/MEF_University_logo.png" alt="MEF University Logo" width="200"/>

MEF UNIVERSITY

<br/><br/><br/>

# AUTOGEN COURSE GENERATOR: AN AUTOMATED AI-POWERED EDUCATIONAL CONTENT CREATION AND VIDEO GENERATION PLATFORM

<br/><br/>

**Senior Design Project II**

<br/><br/><br/><br/><br/>

**Polat Korkmaz**

<br/><br/><br/><br/><br/>

**2026**

</div>

<div style="page-break-after: always;"></div>

# Abstract

The **AutoGen Course Generator** is an advanced web-based platform designed to revolutionize the creation of educational content by leveraging the power of Large Language Models (LLMs) and Multi-Agent Systems. In traditional e-learning development, creating high-quality video courses is a labor-intensive process requiring expertise in instructional design, scriptwriting, and video editing. This project automates the entire pipeline: from curriculum planning and script generation to the programmatic rendering of video lectures.

Built on a robust technical stack comprising **FastAPI**, **Google Cloud Platform (GCP)**, and Microsoft's **AutoGen** framework, the system orchestrates a team of specialized AI agents—including Planners, Researchers, Scriptwriters, and Reviewers—to collaborate on course creation. The platform integrates **Retrieval-Augmented Generation (RAG)** using **Qdrant** to ensure content accuracy and relevance. The final video output is synthesized programmatically using **MoviePy**, combining AI-generated audio, visual assets, and text overlays into a cohesive learning experience.

This report details the system's architecture, the design of the multi-agent workflows, the technical implementation of the video rendering engine, and the deployment strategy using **Google Cloud Run** for a scalable, serverless production environment.

**Keywords**: *Artificial Intelligence, Multi-Agent Systems, AutoGen, Automated Video Generation, EdTech, Cloud Computing, Google Cloud Platform.*

<div style="page-break-after: always;"></div>

# Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Definition and Objectives](#2-problem-definition-and-objectives)
3. [Theoretical Background](#3-theoretical-background)
    - 3.1. Large Language Models (LLMs)
    - 3.2. Multi-Agent Systems and AutoGen
    - 3.3. Retrieval-Augmented Generation (RAG)
4. [System Architecture](#4-system-architecture)
    - 4.1. High-Level Overview
    - 4.2. Technology Stack
    - 4.3. Cloud Infrastructure
5. [Multi-Agent System Design](#5-multi-agent-system-design)
    - 5.1. Agent Roles and Responsibilities
    - 5.2. Workflow Orchestration
6. [Video Generation Engine](#6-video-generation-engine)
    - 6.1. Script-to-Video Pipeline
    - 6.2. Programmatic Editing with MoviePy
7. [Implementation Details](#7-implementation-details)
    - 7.1. Database Schema
    - 7.2. Backend API
    - 7.3. Deployment and CI/CD
8. [Proposed User Manual](#8-proposed-user-manual)
9. [Conclusion and Future Work](#9-conclusion-and-future-work)
10. [References](#10-references)

<div style="page-break-after: always;"></div>

# 1. Introduction

The landscape of education is rapidly shifting towards digital and on-demand learning. However, the barrier to entry for creating high-quality educational material remains high. Producing a standard video course involves multiple disparate skills: domain research, curriculum structuring, engaging scriptwriting, voice recording, and complex video editing.

The **AutoGen Course Generator** addresses these challenges by automating the heavy lifting of course creation. By treating the course creation process as a collaborative effort between specialized AI agents, the system simulates a real-world production team. A "Planner" agent outlines the course, a "Researcher" gathers facts, a "Writer" drafts the script, and a "Video Generator" compiles the final media.

This project not only demonstrates the capabilities of modern Generative AI but also provides a practical, scalable solution for educators, corporations, and content creators to rapidly deploy training materials.

# 2. Problem Definition and Objectives

## 2.1 Problem Statement
Manual course creation is:
*   **Time-Consuming**: A single hour of high-quality video content can take 20-50 hours to produce.
*   **Expensive**: Hiring instructional designers, voice actors, and video editors is costly.
*   **Inconsistent**: Maintaining a consistent tone and quality across modules is difficult for individuals.
*   **Static**: Updating video content requires re-recording and re-editing.

## 2.2 Objectives
The primary objective of this project is to build a fully automated end-to-end platform where a user can input a simple topic (e.g., "Introduction to Python") and receive a complete, ready-to-watch video course.

Specific sub-objectives include:
1.  **Intelligent Curriculum Design**: Use AI to generate logical, structured course outlines.
2.  **Content Accuracy**: Implement RAG to ground AI generation in factual documents.
3.  **Automated Video Production**: Eliminate the need for manual video editing software.
4.  **Scalable Deployment**: Ensure the system can handle concurrent users and heavy rendering loads via Cloud technologies.

# 3. Theoretical Background

## 3.1 Large Language Models (LLMs)
The core intelligence of the system is provided by OpenAI's **GPT-4o-mini**, accessed via API. LLMs are deep learning models trained on vast amounts of text data, capable of understanding context, generating human-like text, and performing complex reasoning tasks.

## 3.2 Multi-Agent Systems and AutoGen
While a single LLM prompt can generate text, complex tasks require iterative refinement and specialized roles. **AutoGen** is a framework from Microsoft that enables the development of LLM applications using multiple agents that can converse with each other to solve tasks. In this project, AutoGen is used to create a "Council of Agents" where each agent has a specific persona (e.g., "The Critic") to ensure high-quality output.

## 3.3 Retrieval-Augmented Generation (RAG)
To prevent "hallucinations" (AI making up facts), the system uses **RAG**. Documents (PDFs, text files) are ingested into a **Qdrant** vector database. When an agent needs to write a lesson, it first queries this database for relevant context, which is then fed into the prompt, ensuring the output is grounded in provided reference materials.

# 4. System Architecture

## 4.1 High-Level Overview
The system follows a microservices-based architecture pattern, although currently deployed as a monolithic container for simplicity in managing shared state during development.

1.  **User Interface**: A web-based dashboard where users manage courses.
2.  **API Layer (FastAPI)**: Handles requests, authentication, and orchestrates background jobs.
3.  **Agent Layer**: The AutoGen environment running the conversational workflows.
4.  **Media Engine**: A Python-based rendering pipeline using MoviePy.
5.  **Data Layer**: Cloud SQL (PostgreSQL) for relational data and Qdrant for vector data.
6.  **Storage Layer**: Google Cloud Storage for persistent video hosting.

## 4.2 Technology Stack
*   **Language**: Python 3.10+
*   **Web Framework**: FastAPI
*   **Database**: PostgreSQL 15, SQLAlchemy (ORM), Alembic (Migrations)
*   **Vector DB**: Qdrant
*   **Video Processing**: MoviePy, FFMPEG
*   **AI Framework**: Microsoft AutoGen, OpenAI API
*   **Containerization**: Docker

## 4.3 Cloud Infrastructure
The project is designed for **Google Cloud Platform (GCP)**:
*   **Cloud Run**: chosen for its serverless, scale-to-zero capabilities. It hosts the main application container.
*   **Cloud Build**: provides CI/CD pipelines, automatically building and deploying the app on git push.
*   **Cloud SQL**: A managed PostgreSQL service ensuring data persistence and backup.
*   **Secret Manager**: securely manages API keys (OpenAI, Qdrant).

# 5. Multi-Agent System Design

The `agents/` directory contains the definitions for the specialized agents. The interaction flow is as follows:

## 5.1 Agent Roles
1.  **CoursePlannerAgent**:
    *   *Input*: User topic (e.g., "Machine Learning").
    *   *Role*: Breaks the topic down into Modules and Lessons. Defines learning objectives.
2.  **ResearchAgent**:
    *   *Input*: Lesson topic.
    *   *Role*: Queries the Qdrant vector database to find relevant facts and definitions.
3.  **LessonWriterAgent**:
    *   *Input*: Course outline + Research data.
    *   *Role*: Writes the actual script for the video, including spoken narration and visual cues.
4.  **ReviewAgent**:
    *   *Input*: Draft script.
    *   *Role*: Critiques the script for clarity, flow, and educational value.
5.  **VideoScriptAgent**:
    *   *Input*: Finalized text.
    *   *Role*: Converts the text into a structured JSON format that the `VideoBuilder` can parse (timecodes, visual assets, text overlays).
6.  **VideoGeneratorAgent**:
    *   *Role*: Triggers the rendering pipeline.

## 5.2 Workflow Orchestration
The agents are managed by a central `AgentManager` class. When a request is received, the Manager initializes the required agents and initiates a "Group Chat" or specific sequential handoffs depending on the complexity of the task. For example, the `ReviewAgent` can reject a draft from the `LessonWriterAgent`, triggering a rewrite loop before the content is approved for video generation.

# 6. Video Generation Engine

One of the most technically challenging aspects of this project is the **programmatic video generation** located in `course_material_service/video_builder.py`.

## 6.1 Script-to-Video Pipeline
The engine receives a structured JSON script containing segments. Each segment defines:
*   **Narration**: Text to be converted to speech (TTS).
*   **Visuals**: Background colors, images, or code snippets.
*   **Duration**: Calculated based on the length of the audio.

## 6.2 Programmatic Editing with MoviePy
The system uses the `MoviePy` library to compose the video frame by frame:
1.  **Audio Synthesis**: The text narration is sent to an OpenAI TTS endpoint to generate high-quality audio files (`.mp3`).
2.  **Visual Composition**:
    *   **TextClips**: Titles and bullet points are rendered with specific fonts and colors.
    *   **ImageClips**: Relevant stock images or diagrams are fetched and resized.
    *   **Transitions**: Cross-fades and slides are applied between segments to ensure a smooth viewing experience.
3.  **Rendering**: The audio and visual tracks are composited. FFMPEG is used under the hood to encode the final output into an optimized `.mp4` format (H.264 codec).
4.  **Storage**: The final binary is uploaded to Google Cloud Storage to ensure persistence, as the Cloud Run filesystem is ephemeral.

# 7. Implementation Details

## 7.1 Database Schema
The relational database (PostgreSQL) manages the structured data:
*   **Users**: Authentication and profile data.
*   **Courses**: Metadata (Title, Description) and ownership.
*   **Modules**: Groupings of lessons.
*   **Lessons**: The core unit, containing the generated script and links to the final video URL.
*   **Enrollments**: Tracking which users are taking which courses.

## 7.2 Backend API
FastAPI was chosen for its performance (async support) and automatic documentation (Swagger UI).
*   **Endpoint**: `/api/v1/courses/generate` - Triggers the Agent workflow.
*   **Endpoint**: `/api/v1/video/stream/{video_id}` - Proxies or redirects to the cloud storage URL for playback.
*   **Background Tasks**: Since video generation is CPU-intensive and slow, it is handled asynchronously to prevent blocking the HTTP interface.

## 7.3 Deployment and CI/CD
The project utilizes a modern **DevOps** pipeline defined in `cloudbuild.yaml`:
1.  Code is pushed to GitHub.
2.  Cloud Build triggers automatically.
3.  It builds the Docker image from `Dockerfile`.
4.  It runs unit tests (PyTest).
5.  On success, it pushes the image to Artifact Registry and deploys a new revision to Cloud Run.
6.  Database migrations (`alembic upgrade head`) are applied automatically to keep the schema in sync.

# 8. Proposed User Manual

## 8.1 Getting Started
1.  Navigate to the deployed URL.
2.  Create an account or log in.
3.  On the Dashboard, click **"Create New Course"**.

## 8.2 Creating a Course
1.  Enter a detailed topic (e.g., "Advanced Java Concurrency").
2.  Select the target audience level (Beginner, Intermediate, Advanced).
3.  (Optional) Upload reference PDF materials for the AI to study.
4.  Click **"Generate Outline"**. The AI planner will propose a structure.
5.  Approve or specificy modifications to the outline.

## 8.3 Generating Video
1.  Select a lesson from the outline.
2.  Click **"Generate Video"**.
3.  Wait for the progress bar (approx. 2-5 minutes per lesson).
4.  Once complete, the video player will appear. You can watch, download, or regenerate the content if desired.

# 9. Conclusion and Future Work

The **AutoGen Course Generator** successfully demonstrates the potential of AI in automating complex creative workflows. By orchestrating specialized agents, the system achieves a level of quality and coherence that is difficult to achieve with simple, single-prompt LLM interactions. The integration of MoviePy allows for truly dynamic media creation without human intervention.

## Future Work
*   **Avatars**: Integration of AI video avatars (e.g., HeyGen or localized lipsync) for a more human-like presenter.
*   **Interactive Quizzes**: Extending the `QuizAgent` to generate interactive HTML5 overlays on top of the video.
*   **Voice Cloning**: Allowing users to clone their own voice for the narration.
*   **Feedback Loop**: Implementing a system where users can edit the generated script manually before video rendering, allowing for a "Human-in-the-Loop" workflow.

# 10. References
1.  *Wu, Q., et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications." Microsoft Research.*
2.  *OpenAI (2024). "GPT-4 Technical Report."*
3.  *Google Cloud (2024). "Serverless Computing with Cloud Run."*
4.  *MoviePy Documentation. "Programmatic Video Editing in Python."*
