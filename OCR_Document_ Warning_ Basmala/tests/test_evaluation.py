
import pytest

from src.utils.evaluation import (
    normalize_text,
    calculate_cer,
    calculate_wer,
    calculate_corpus_cer,
    calculate_corpus_wer,
)


def test_normalize_text():
    assert normalize_text("HELLO   World!") == "hello world"

    # Arabic digits
    assert normalize_text("٥٠٠") == "500"

    # Persian digits
    assert normalize_text("۵۰۰") == "500"


def test_cer_identical_text():
    assert calculate_cer("Paracetamol", "Paracetamol") == 0.0


def test_cer_one_character_error():
    assert calculate_cer("cat", "bat") == pytest.approx(1 / 3)


def test_wer_identical_text():
    assert calculate_wer("hello world", "hello world") == 0.0


def test_wer_one_word_error():
    assert calculate_wer("hello world", "hello there") == pytest.approx(1 / 2)


def test_empty_reference():
    assert calculate_cer("", "") == 0.0
    assert calculate_cer("", "abc") == 1.0

    assert calculate_wer("", "") == 0.0
    assert calculate_wer("", "hello") == 1.0


def test_corpus_metrics():
    pairs = [
        ("cat", "cat"),
        ("dog", "dig"),
    ]

    assert calculate_corpus_cer(pairs) == pytest.approx(1 / 6)
    assert calculate_corpus_wer(pairs) == pytest.approx(1 / 2)
