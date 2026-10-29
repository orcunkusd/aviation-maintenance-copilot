import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import saved_answers as sa

ITEM = {"question": "What is the minimum rivet spacing?", "answer": "x [1]", "sources": [{"source": "a.pdf", "page": 1, "content": "c"}]}


def test_find_ignores_case_spacing_and_question_mark():
    assert sa.find([ITEM], "  what is the MINIMUM rivet   spacing ") is ITEM
    assert sa.find([ITEM], "What is the minimum rivet spacing?") is ITEM


def test_find_returns_none_for_other_questions():
    assert sa.find([ITEM], "How do I inspect a propeller?") is None
    assert sa.find([], "anything") is None


def test_load_handles_missing_and_broken_files(tmp_path):
    assert sa.load(tmp_path / "nope.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert sa.load(bad) == []


def test_load_drops_incomplete_entries(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([ITEM, {"question": "q", "answer": "", "sources": []}]))
    assert sa.load(p) == [ITEM]
