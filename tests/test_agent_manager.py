from typing import Any, Dict

from manager.agent_manager import AgentManager


def test_agent_manager_integration(monkeypatch):
    manager = AgentManager()

    def fake_generate_course_plan(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": "Demo Course",
            "modules": [{"title": "Module 1", "lessons": ["Lesson 1"]}],
        }

    def fake_generate_lesson(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": input_json["lesson_name"],
            "text": "This is a sufficiently long lesson text.",
            "summary": "Summary",
        }

    def fake_generate_script(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "lesson": input_json["title"],
            "scenes": [{"text": "Scene 1", "duration": 10}],
        }

    def fake_generate_quiz(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "lesson": input_json["title"],
            "questions": [
                {"question": "Q1", "options": ["a", "b", "c"], "correct_index": 1},
                {"question": "Q2", "options": ["a", "b", "c"], "correct_index": 2},
                {"question": "Q3", "options": ["a", "b", "c"], "correct_index": 0},
            ],
        }

    def fake_generate_video(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        return {"video_path": "generated_videos/lesson_video.mp4"}

    def fake_validate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok", "validated_content": input_json, "issues": []}

    monkeypatch.setattr(manager.course_planner, "generate", fake_generate_course_plan)
    monkeypatch.setattr(manager.lesson_writer, "generate", fake_generate_lesson)
    monkeypatch.setattr(manager.video_script_agent, "generate", fake_generate_script)
    monkeypatch.setattr(manager.quiz_agent, "generate", fake_generate_quiz)
    monkeypatch.setattr(manager.video_generator, "generate", fake_generate_video)
    monkeypatch.setattr(manager.validator, "generate", fake_validate)

    result = manager.run(["Outcome 1"])
    assert "course_plan" in result
    assert result["lessons"]
    assert result["videos"]
    assert result["quizzes"]

