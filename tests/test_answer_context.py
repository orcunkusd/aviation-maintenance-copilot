from answer import SYSTEM, build_context


def _hit(n):
    return {"id": f"id{n}", "text": f"text {n}", "meta": {"source": "AC.pdf", "page": n}}


def test_context_numbers_passages_from_one():
    ctx = build_context([_hit(7), _hit(9)])
    assert ctx.startswith("[1] (source: AC.pdf, page 7)\ntext 7")
    assert "\n\n[2] (source: AC.pdf, page 9)\ntext 9" in ctx


def test_empty_hits_give_empty_context():
    assert build_context([]) == ""


def test_system_prompt_enforces_citations_and_refusal():
    assert "square brackets" in SYSTEM
    assert "could not find it" in SYSTEM
    assert "Do not use outside knowledge" in SYSTEM
