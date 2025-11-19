from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    name = "research_agent"

    def generate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conducts initial research/brainstorming for the course topic.
        Input: {"learning_outcomes": [...]}
        Output: {"context": "...", "key_concepts": [...], "target_audience_analysis": "..."}
        """
        learning_outcomes = input_json.get("learning_outcomes", [])
        outcomes_str = "\n".join(f"- {o}" for o in learning_outcomes)

        prompt = f"""
        You are an expert educational researcher. Your goal is to prepare a comprehensive research briefing for a course creator.
        
        The course has the following learning outcomes:
        {outcomes_str}

        Please provide a detailed research summary that includes:
        1. **Key Concepts**: A list of core concepts that must be covered to achieve these outcomes.
        2. **Target Audience Analysis**: Who is this course for? What are their likely pain points?
        3. **Real-World Applications**: Examples and case studies that can be used.
        4. **Common Misconceptions**: What do students often get wrong about this topic?

        Return your response as a JSON object with the following structure:
        {{
            "key_concepts": ["concept1", "concept2", ...],
            "target_audience_analysis": "detailed text...",
            "real_world_applications": ["app1", "app2", ...],
            "common_misconceptions": ["misc1", "misc2", ...],
            "summary": "A brief executive summary of the research."
        }}
        """

        llm_result = self.call_llm(prompt)
        return self.validate_json(llm_result)
