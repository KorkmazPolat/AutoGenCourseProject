from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import BaseAgent


class ReviewAgent(BaseAgent):
    name = "review_agent"

    def generate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reviews a piece of content (plan, lesson, or script) and provides feedback.
        Input: {"content_type": "plan|lesson|script", "content": {...}, "context": {...}}
        Output: {"score": 0-10, "feedback": "...", "approved": bool}
        """
        content_type = input_json.get("content_type", "content")
        content = input_json.get("content", {})
        context = input_json.get("context", {})

        prompt = f"""
        You are a strict educational quality assurance specialist. You are reviewing a {content_type}.

        **Context (Research/Background):**
        {context}

        **Content to Review:**
        {content}

        Evaluate the content based on:
        1. Alignment with learning outcomes.
        2. Clarity and depth.
        3. Engagement and flow.
        4. Accuracy.

        Provide a score from 0 to 10 (where 10 is perfect).
        If the score is 8 or higher, set "approved" to true. Otherwise, false.
        Provide specific, actionable feedback for improvement.

        Return JSON:
        {{
            "score": <int>,
            "feedback": "<string>",
            "approved": <bool>
        }}
        """

        llm_result = self.call_llm(prompt)
        return self.validate_json(llm_result)
