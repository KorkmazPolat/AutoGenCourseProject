# AutoGen Course Generator - 10 Minute Presentation Script

**Total Estimated Time:** ~10:00 minutes
**Speakers:** [Decide who speaks which parts, e.g., Polat (Tech), Mert (Frontend/Demo), Sinem (Backend/QA)]

---

## 1. Introduction (Slide 1)
**Time:** 0:00 - 0:45 (45 seconds)

**Speaker:**
"Good afternoon everyone. Welcome to our Senior Design Project presentation. We are presenting the **AutoGen Course Generator**."

"We are a team of three: I am **Polat Korkmaz**, covering the core Architecture and Research. With me today are **Mert Kırtı**, our Backend Developer, and **Sinem Özbey**, our Frontend Developer."

"Our project addresses a simple question: What if you could create a high-quality video course just by typing a single sentence? We have built an automated platform that uses **Multi-Agent Systems** to act as a production studio in the cloud, converting simple text prompts into full educational video content."

---

## 2. Purpose of the Project (Slide 2)
**Time:** 0:45 - 1:30 (45 seconds)

**Speaker:**
"So, why did we build this?"

"First, to **Democratize Education**. There are brilliant experts out there who can teach but don't know how to edit video or manage audio. We want to remove that technical barrier."

"Second, **Acceleration**. We want to reduce production time from weeks to minutes."

"And finally, to showcase **Agentic AI**. We moved beyond simple chatbots. We wanted to prove that autonomous agents can coordinate, plan, critique, and execute complex workflows without human intervention."

---

## 3. The EdTech Bottleneck (Slide 3)
**Time:** 1:30 - 2:30 (1 minute)

**Speaker:**
"Let's look at the problem we are solving today. The shift to digital learning is massive, but the production pipeline is completely archaic."

"We call this the **'Golden Ratio of Production'**: Use your mouse to point at **50:1**. It currently takes about **50 hours** of human labor to produce just **1 hour** of high-quality finished courseware."

"You need a scriptwriter, a researcher to verify facts, a voice actor, and a video editor. At market rates, a single course can cost upwards of **$830**. This high cost and slow speed is the bottleneck that prevents real-time, personalized education."

---

## 3.5. Existing Solutions & Literature (Slide 3.5)
**Time:** 2:30 - 3:00 (30 seconds)

**Speaker:**
"Before we built our system, we analyzed existing solutions in the literature."

*   "**Standard LLMs (ChatGPT):** Good for text, but they hallucinate facts and can't produce video."
*   "**Template Tools (Canva/Vyond):** Good for visuals, but they are 'dumb'. They have no logic and still require manual drag-and-drop."

"Our approach is novel because it combines **Agentic Reasoning** with **Code Execution**."

---

## 4. Infrastructure Overview (Slide 4)
**Time:** 3:00 - 3:45 (45 seconds)

**Speaker:**
"To support this agentic workflow, we built a Cloud-Native architecture. Here is **why** we chose these specific technologies:"

*   "**Cloud Run:** Chosen for **Cost Efficiency**. It scales to zero when idle, so we don't pay for empty servers."
*   "**FastAPI:** Chosen for **Async Support**. Agents need to talk concurrently, and Python's asyncio handles this perfectly."
*   "**Qdrant:** Chosen for **Semantic RAG**. We need to find concepts, not just keywords."

---

## 5. Hybrid Database Design (Slide 5)
**Time:** 3:45 - 4:30 (45 seconds)

**Speaker:**
"Data is the heart of our system, and we use a **Hybrid Design**."

"On the left, we have our trusted **Relational Database** (PostgreSQL). This handles structured business logic: Users, Course Metadata, and Progress tracking. It's reliable and strictly typed."

"On the right, we have **Qdrant**, our Vector Database. This handles unstructured 'knowledge'. We embed textbooks and papers into 1536-dimensional vectors. This allows our agents to search for *concepts* rather than just keywords, enabling semantic retrieval for accurate content generation."

---

## 6. Deep Dive: Phase 1 - Workflow (Slide 6)
**Time:** 4:30 - 5:15 (45 seconds)

**Speaker:**
"Now, let's look 'under the hood' at our Agentic Brain. We use **Microsoft AutoGen**."

"Instead of a single AI trying to do everything, we created a **Group Chat**.
On the left is the **User Proxy**—that's the user's request.
On the right is the **GroupChatManager**. It acts like a project manager."

"When a request comes in, the Manager doesn't just answer. It delegates. It might say, 'Planner, outline this.' Then 'Researcher, verify these facts.' Then 'Writer, draft the script.' The agents talk to *each other* until the task is complete."

---

## 7. Deep Dive: Phase 2 - The Squad (Slide 7)
**Time:** 5:15 - 6:00 (45 seconds)

**Speaker:**
"Let's meet the squad. Each agent has a distinct 'System Message' acting as its persona."

"The **Course Planner** is our Architect. It outputs structured JSON outlines."
"The **Researcher** is the Fact Checker. It uses RAG to pull citations."
"The **Script Writer** is the Storyteller, focusing on engagement."
"The **Critic** is Quality Control. If a script is boring, the Critic *rejects* it and forces a rewrite before the user ever sees it."
"Finally, the **Video Producer** acts as the engineer, converting the text script into code instructions for video rendering."

---

## 8. Deep Dive: Phase 3 - RAG (Slide 8)
**Time:** 6:00 - 6:45 (45 seconds)

**Speaker:**
"One major issue with AI is hallucinations. To prevent this, we built a **RAG (Retrieval-Augmented Generation)** engine."

"We take source documents—PDFs, wikis, books—and convert them into embeddings using OpenAI's embedding models. These are stored in Qdrant."

"When the Researcher agent needs to write about, say, 'Quantum Mechanics', it queries the database. Qdrant returns the exact paragraphs from the textbooks, and we feed those into the agent's context window. This ensures 100% factual accuracy."

---

## 9. How it Works: Video Engine (Slide 9)
**Time:** 6:45 - 7:30 (45 seconds)

**Speaker:**
"Phase 4 is the **Video Engine**. This is where the magic happens."

"We don't use cameras. We use code.
1. We take the **JSON script** from the agents.
2. We generate **Audio** using high-definition Text-To-Speech.
3. We use **MoviePy**, a Python library, to programmatically assemble the video. We synchronize the audio duration with images and text overlays."
4. Finally, we render it out as an **MP4** and upload it to the cloud."

---

## 10. Live Demos (Slides 10, 11, 12, 13)
**Time:** 7:30 - 9:00 (1 minute 30 seconds)

**Speaker:**
*(Breeze through slides quickly)*
"Let's see it in action."

**(Slide 10)** "Here is our **Course Generator Form**. You simply input 'System Design for Beginners', select your audience, and click Generate. The agents spin up instantly."

**(Slide 11)** "We also built a **Drag & Drop Builder**. The AI gives you a starting point, but we believe in 'Human-in-the-loop'. You can drag new modules, edit content, and rearrange the flow manually."

**(Slide 12 & 13)** "Once generated, the course lives in the **Library**. We also have individual tools for specific tasks, like generating a single slide deck or a quick quiz."

---

## 11. Cost Analysis (Slide 14)
**Time:** 9:00 - 9:40 (40 seconds)

**Speaker:**
"The impact is financial."
"Remember the **$830** traditional cost for a 10-minute module?"

"With our AutoGen platform:
- Gemini tokens cost 2 cents.
- TTS costs 36 cents.
- Compute costs 5 cents."

"**Total: $0.44.** That is a **99.95% reduction in cost** and a **68x increase in speed**. This changes the economics of education entirely."

---

## 12. Future & Closing (Slides 15 & 16)
**Time:** 9:40 - 10:00 (20 seconds)

**Speaker:**
"Looking ahead to 2026, we plan to integrate **AI Avatars** via HeyGen and **Voice Cloning** for personalized delivery.

"Thank you for listening. Our project is live at the link on the screen. We are happy to take your questions."
