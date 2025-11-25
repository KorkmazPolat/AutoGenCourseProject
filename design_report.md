# Design Update Report: Agentic Course Generator

**Date:** November 21, 2025
**Project:** AutoGen Course Project
**Status:** Active Development

## 1. Executive Summary
This report outlines the recent design implementations and architectural enhancements made to the Agentic Course Generator platform. Key updates include a complete relational database schema implementation, significant upgrades to the Visual Course Builder interface, and enhanced system architecture visualizations.

## 2. Database Architecture
We have transitioned to a robust, asynchronous relational database system to support a multi-user environment.

*   **Technology Stack:** SQLite with `aiosqlite` and SQLAlchemy (Async).
*   **Core Schema:**
    *   **User Management:** `User` table handling credentials, profiles, and roles (Admin/Student).
    *   **Course Structure:** Hierarchical design with `Course` -> `CourseModule` -> `Lesson`.
    *   **Content Assets:** `LessonAsset` table for storing generated content like video scripts, quizzes, and audio files.
    *   **Progress Tracking:** `Enrollment`, `LessonCompletion`, and `QuizAttempt` tables to track student progress and performance.
*   **Key Features:**
    *   Fully asynchronous database sessions for high-performance API handling.
    *   Automatic table initialization on startup.
    *   Foreign key constraints ensuring data integrity across the course hierarchy.

## 3. Visual Course Builder (UI/UX)
The Course Builder (`course_builder.html`) has received major usability upgrades to facilitate intuitive course design.

*   **Drag-and-Drop Interface:** Users can now drag course components (Modules, Lessons) onto a canvas.
*   **Persistent Connections:**
    *   **Visual Linking:** Arrows now persist between elements to define the learning path.
    *   **Dynamic Updates:** Connections automatically redraw and update when elements are moved.
    *   **Ordering:** Connections display order numbers (e.g., "1", "2") to clearly indicate the sequence of lessons.
*   **Interaction Design:**
    *   **Deletion:** Users can select and delete specific connections.
    *   **State Management:** The visual graph state is maintained to ensure the visual representation matches the logical flow.

## 4. System Architecture Visualization
To better understand the agentic workflow, we have enhanced the `system_architecture.html` page.

*   **Agent Workflow:** Visual representation of how the `CourseManager`, `ContentAgent`, and `VideoAgent` interact.
*   **Live Diagrams:** Integration of dynamic diagrams (Mermaid/SVG) to show the flow of data from user prompt to final course generation.
*   **Status Monitoring:** UI elements designed to show the real-time status of background generation tasks.

## 5. Use Case Analysis
The `use_cases.html` page has been expanded to document the system's capabilities.

*   **Detailed Scenarios:** Added comprehensive descriptions for key user stories (e.g., "Instructor creates a Python course", "Student takes a quiz").
*   **UML Diagrams:** Added detailed UML Class Diagrams to visualize the database structure and object relationships directly within the application documentation.

## 6. Next Steps & Recommendations
*   **Backend Integration:** Fully connect the Visual Course Builder's JSON output to the `generate_course` backend endpoint.
*   **Authentication:** Implement the API endpoints for User Registration and Login using the new `User` model.
*   **Video Pipeline:** Finalize the integration of `MoviePy` for automated video generation, ensuring compatibility with the new asset database.
