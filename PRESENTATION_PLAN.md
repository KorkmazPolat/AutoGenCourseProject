# AutoGen Course Generator - Presentation Plan

This document outlines the slide deck for the "AutoGen Course Generator" project presentation.
**Total Duration**: 10 Minutes
**Speaker**: Polat Korkmaz
**Visual System (Strict Consistency)**: **"Modern Tech Efficiency"**
*   **Palette**:
    *   **Background**: Unified "Paper White" (#F8F9FA) for ALL slides. No dark modes.
    *   **Accents**: Deep Navy (#1E293B) for primary text, Vibrant Blue (#3B82F6)-to-Pink (#EC4899) gradients for highlights/graphics.
*   **Typography**: Inter or SF Pro Display (Clean Sans-Serif).
*   **Components**: Softly rounded cards, frosted glass effects, subtle drop shadows.
*   **Layout Rule**: Every slide must follow a clean, grid-based layout with ample whitespace.

---

## 1. Opening Slide (0:00 - 0:30)
*   **Slide Type**: Title / Introduction.
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Graphic**: A large, abstract, multi-colored gradient shape (Blue to Purple to Pink) flowing organically across the bottom right corner.
    *   **Text Layout**: Left-aligned, heavy typography.
        *   **Logo**: MEF University Logo (Color) top-left.
        *   **Title**: **"AutoGen Course Generator"** in huge Deep Navy text.
        *   **Subtitle**: "Democratizing Education with Multi-Agent AI" in Slate Grey.
        *   **Footer**: "Polat Korkmaz | Senior Design Project II | 2026".
*   **Speaker Notes**:
    *   "Good morning/afternoon. I am Polat Korkmaz."
    *   "Today, I present the **AutoGen Course Generator**: A fully automated platform that takes a simple user idea and architects a complete educational ecosystem from scratch."

---

## 2. Core Problem Integration (0:30 - 1:00)
*   **Slide Type**: Narrative / Problem Statement.
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Layout**: Asymmetrical 2-Column Layout.
    *   **Left Column**: Bold vertical bar in Deep Navy with text **"The Bottleneck"**.
    *   **Right Column**: Three floating "Glass Cards" loaded vertically with soft shadows.
        *   *Card 1*: **"Cost"** - Icon of a burning wallet ($500+/min).
        *   *Card 2*: **"Time"** - Icon of a calendar flipping (Weeks of work).
        *   *Card 3*: **"Consistency"** - Icon of a jagged graph.
    *   **Accent**: A "Red Stamp" effect animating onto the cards saying "MANUAL EFFORT".
*   **Speaker Notes**:
    *   "Creating high-quality courseware is expensive and slow. It prevents great ideas from becoming shared knowledge."

---

## 3. The Platform Features (1:00 - 4:00)
*(Showcasing the Website Modules & User Experience)*

### Slide 3.1: The Course Builder (From Idea to Structure)
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Header**: **"Intelligent Course Builder"** in Deep Navy.
    *   **Graphic**: Central Flow Diagram on white.
        *   *Input*: Clean form field UI showing "Topic: Python".
        *   *Connector*: Animated Blue-to-Pink gradient line.
        *   *Output*: Clean hierarchical tree nodes showing "Modules & Lessons".
    *   **Style Note**: Use clean outlines for icons, no heavy dark fills.
*   **Speaker Notes**:
    *   "It starts with the **Course Builder**. The user simply fills out a form defining the topic and audience."
    *   "Behind the scenes, the AI Architect takes this simple input and constructs a comprehensive curriculum tree, breaking the topic down into logical modules and lessons."

### Slide 3.2: The Presentation Tool
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Header**: **"Instant Slide Generator"** in Deep Navy.
    *   **Graphic**: High-fidelity UI Mockup floating in center with soft shadow.
        *   Show the "Loading..." states: *Analysing -> Structuring -> Designing*.
        *   Overlay a semi-transparent code block showing valid JSON generation.
    *   **Badge**: "Powered by Gemini 2.0" pill badge in gradient color.
*   **Speaker Notes**:
    *   "Lectures need visuals. Our **Presentation Tool** uses Google's Gemini model to instantly generate professional slide decks."
    *   "It doesn't just write text; it designs the layout and speaker notes in real-time."

### Slide 3.3: The Reading Tool
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Header**: **"Deep-Dive Reading Engine"** in Deep Navy.
    *   **Graphic**: Split Card Layout.
        *   *Left Card*: Icon of a Brain/AI processing text.
        *   *Right Card*: Beautifully typeset "Article View" mockup with clean headers.
    *   **Accent**: Small "Qdrant/RAG" verified shield icon.
*   **Speaker Notes**:
    *   "Not everyone learns by watching. The **Reading Tool** generates comprehensive written guides and textbooks for each module."
    *   "It uses RAG to ensure the written content is factually accurate and rich in detail."

### Slide 3.4: The Quiz Tool
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Header**: **"Adaptive Assessment (Quiz Tool)"** in Deep Navy.
    *   **Graphic**: Interactive UI Mockup.
        *   Question card: "What is the output of print(2+2)?"
        *   Selection state: "4" highlighted in Green.
    *   **Keywords**: Clean floating pills: "Verification", "Active Recall", "Auto-Grading".
*   **Speaker Notes**:
    *   "Finally, to verify learning, the **Quiz Tool** automatically generates assessments based on the course content."
    *   "It creates questions, validates answers, and provides immediate feedback to the student."

---

## 4. Technical Architecture: Under the Hood (4:00 - 7:00)
*(Explaining the Tech Stack that powers features above)*

### Slide 4.1: The Architect (Microsoft AutoGen)
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Header**: **"The Brain: Multi-Agent Orchestration"**.
    *   **Center Graphic**: "Roundtable" visualization using clean, circular portraits for agents.
        *   Colors: Each agent circle uses a soft pastel background (Blue, Green, Purple).
        *   Lines: Thin, elegant connecting lines representing conversation.
    *   **Text**: Minimal text, strict hierarchy.
*   **Speaker Notes**:
    *   "How do we coordinate all these tools? We use **Microsoft AutoGen**."
    *   "A 'Council of Agents' collaborates. The Planner structures the course, and the Researcher verifies the facts."

### Slide 4.2: The Knowledge Engine (RAG with Qdrant)
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Header**: **"The Memory: Grounded Reality"**.
    *   **Graphic**: Linear Process Flow (Left to Right).
        *   Document Icon -> Gradient Funnel (Vectorization) -> DB Cylinder (Qdrant) -> Search Icon.
    *   **Palette**: Use Teal/Emerald gradients for this slide to signify "Truth/Accuracy", but keep white background.
*   **Speaker Notes**:
    *   "To prevent hallucinations, we use **Qdrant** for RAG. Agents queries uploaded documents to ensure accuracy."

### Slide 4.3: The Production Studio (MoviePy)
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Header**: **"The Studio: Programmatic Video"**.
    *   **Graphic**: "Exploded Layers" Isometric View.
        *   Base Layer: Code Snippet (Python).
        *   Middle Layer: Audio Waveform.
        *   Top Layer: Final Video Frame.
    *   **Tech Badges**: Small logos for MoviePy & FFMPEG at bottom right.
*   **Speaker Notes**:
    *   "For the final video output, we use **MoviePy**. We write code that *is* the video editor, programmatically stitching audio and visuals together."

---

## 5. Deployment & Infrastructure (7:00 - 8:00)
*   **Slide Type**: Infrastructure Diagram.
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Graphic**: Isometric 3D Cloud Map.
        *   Clean white/grey blocks for Cloud Run, Cloud SQL, Cloud Build.
        *   Connected by Blue animated pipes.
    *   **Text**: Labels in Slate Grey typography.
*   **Speaker Notes**:
    *   "We deployed this as a cloud-native application on GCP. Using Cloud Run, it scales to zero when not in use."

---

## 6. Future Roadmap (8:00 - 9:00)
*   **Slide Type**: Timeline.
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Graphic**: Horizontal Timeline Line (Blue Gradient).
    *   **Milestones**: Three circular nodes on the line.
        *   *Node 1*: **"AI Avatars"** (Thumbnail of face).
        *   *Node 2*: **"Interactive Quizzes"** (Thumbnail of checkbox).
        *   *Node 3*: **"Voice Cloning"** (Thumbnail of wave).
*   **Speaker Notes**:
    *   "The next phase introduces hyper-realistic AI Avatars and voice cloning."

---

## 7. Closing (9:00 - 10:00)
*   **Slide Type**: Contact / Q&A.
*   **Visual Design**:
    *   **Adhering to Master Style**: White background (#F8F9FA).
    *   **Center**:
        *   **Title**: **"Thank You"** in large Deep Navy text.
        *   **Subtitle**: "Any Questions?"
        *   **Contact**: "polat.korkmaz@mef.edu.tr" in clean sans-serif.
    *   **Graphic**: The same abstract gradient shape from Slide 1, but now in the Top Left corner (closing the loop).
*   **Speaker Notes**:
    *   "Thank you for your time. The future of education is automated and accessible. Questions?"
