from validators.lesson_validator import validate_lesson
from validators.plan_validator import validate_plan
from validators.quiz_validator import validate_quiz
from validators.script_validator import validate_script


def test_validate_plan_ok():
    ok, issues = validate_plan(
        {"modules": [{"title": "M1", "lessons": ["L1", "L2"]}]}
    )
    assert ok
    assert issues == []


def test_validate_lesson_errors():
    ok, issues = validate_lesson({"title": "Hi", "text": "short"})
    assert not ok
    assert issues


def test_validate_script_errors():
    ok, issues = validate_script({"scenes": [{"text": "t", "duration": 0}]})
    assert not ok
    assert issues


def test_validate_quiz_min_questions_and_index():
    ok, issues = validate_quiz(
        {
            "questions": [
                {"question": "Q1", "options": ["a", "b"], "correct_index": 1},
                {"question": "Q2", "options": ["a", "b"], "correct_index": 0},
            ]
        }
    )
    assert not ok
    assert issues

