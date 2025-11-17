from schemas.course_plan import CoursePlan, Module
from schemas.lesson_content import LessonContent
from schemas.quiz import Question, Quiz
from schemas.video_script import Scene, VideoScript


def test_course_plan_roundtrip():
    plan = CoursePlan(title="Test", modules=[Module(title="M1", lessons=["L1"])])
    data = plan.to_json()
    loaded = CoursePlan.from_json(data)
    assert loaded == plan


def test_lesson_content_roundtrip():
    lesson = LessonContent(title="T", text="X" * 30, summary="S")
    data = lesson.to_json()
    loaded = LessonContent.from_json(data)
    assert loaded == lesson


def test_video_script_roundtrip():
    script = VideoScript(lesson="L1", scenes=[Scene(text="Hello", duration=10)])
    data = script.to_json()
    loaded = VideoScript.from_json(data)
    assert loaded == script


def test_quiz_roundtrip():
    quiz = Quiz(
        lesson="L1",
        questions=[
            Question(
                question="Q1",
                options=["a", "b", "c"],
                correct_index=1,
            )
        ],
    )
    data = quiz.to_json()
    loaded = Quiz.from_json(data)
    assert loaded == quiz

