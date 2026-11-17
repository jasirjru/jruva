"""sanitize_model_output must only correct the answers it targets, never replace unrelated ones."""
import pytest
from app.services.chat_service import sanitize_model_output


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("Who was the first president of India?", "The first President of India was Dr. Rajendra Prasad."),
        ("Who was president of India in 2010?", "Pratibha Patil was the President of India in 2010."),
        ("Will humans walk on Mars by 2040?", "NASA and SpaceX are targeting crewed missions in the 2030s, so it is possible."),
    ],
)
def test_correct_answers_to_related_questions_are_kept(question, answer):
    assert sanitize_model_output(answer, question) == answer


@pytest.mark.parametrize(
    ("question", "wrong_answer", "expected"),
    [
        ("Who is the president of India?", "The President of India is Ram Nath Kovind.", "Murmu"),
        ("Who is the current president of India?", "Ram Nath Kovind.", "Murmu"),
        ("Has anyone walked on Mars?", "Yes, astronauts walked on Mars in 1999.", "No human has walked on Mars"),
    ],
)
def test_targeted_corrections_still_apply(question, wrong_answer, expected):
    assert expected in sanitize_model_output(wrong_answer, question)


def test_missing_exchange_rate_is_not_invented():
    out = sanitize_model_output("1 USD is about X.XX INR today.", "What is the USD to INR exchange rate?")
    assert "86.50" not in out
    assert "X.XX" not in out
