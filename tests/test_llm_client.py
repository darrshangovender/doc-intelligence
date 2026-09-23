"""Tests for the LLM client helpers — StubLLM routing and JSON response parsing."""

from __future__ import annotations

import json

import pytest

from doc_intelligence.llm_client import StubLLM, parse_json_response


class TestStubLLMRouting:
    def test_matching_needle_selects_its_response(self) -> None:
        llm = StubLLM(mapping={"INV-42": '{"invoice_no": "INV-42"}'})
        out = llm.complete(system="s", user="text mentioning INV-42 here")
        assert json.loads(out)["invoice_no"] == "INV-42"

    def test_match_is_case_insensitive(self) -> None:
        llm = StubLLM(mapping={"CHECKERS HYPER": '{"merchant": "CHECKERS HYPER"}'})
        assert "merchant" in llm.complete(system="s", user="...checkers hyper...")

    def test_earliest_needle_wins_over_one_cited_later(self) -> None:
        # The bug this pins: a Statement of Work opens with its own title but
        # references the MSA it hangs off two lines down. Dict-order matching
        # returned the MSA fixture and silently gave callers the wrong
        # document's data.
        doc = (
            "STATEMENT OF WORK NO. 1\n\n"
            "Pursuant to the Master Services Agreement dated 15 March 2025, ...\n"
        )
        llm = StubLLM(
            mapping={
                "MASTER SERVICES AGREEMENT": '{"contract_type": "MSA"}',
                "STATEMENT OF WORK NO. 1": '{"contract_type": "SOW"}',
            }
        )
        assert json.loads(llm.complete(system="s", user=doc))["contract_type"] == "SOW"

    def test_result_is_independent_of_mapping_insertion_order(self) -> None:
        doc = "STATEMENT OF WORK NO. 1\nunder the MASTER SERVICES AGREEMENT\n"
        pairs = [
            ("MASTER SERVICES AGREEMENT", '{"t": "MSA"}'),
            ("STATEMENT OF WORK NO. 1", '{"t": "SOW"}'),
        ]
        forward = StubLLM(mapping=dict(pairs)).complete(system="s", user=doc)
        reverse = StubLLM(mapping=dict(reversed(pairs))).complete(system="s", user=doc)
        assert forward == reverse

    def test_longer_needle_wins_a_positional_tie(self) -> None:
        llm = StubLLM(
            mapping={
                "MASTER SERVICES": '{"t": "short"}',
                "MASTER SERVICES AGREEMENT": '{"t": "long"}',
            }
        )
        out = llm.complete(system="s", user="MASTER SERVICES AGREEMENT follows")
        assert json.loads(out)["t"] == "long"

    def test_falls_back_to_default_when_nothing_matches(self) -> None:
        llm = StubLLM(mapping={"INV-42": "{}"}, default='{"fallback": true}')
        assert json.loads(llm.complete(system="s", user="unrelated"))["fallback"] is True

    def test_raises_when_no_match_and_no_default(self) -> None:
        llm = StubLLM(mapping={"INV-42": "{}"})
        with pytest.raises(RuntimeError, match="no matching response"):
            llm.complete(system="s", user="unrelated")

    def test_responder_takes_precedence_over_mapping(self) -> None:
        llm = StubLLM(
            responder=lambda system, user: f"saw:{user}",
            mapping={"INV-42": "{}"},
        )
        assert llm.complete(system="s", user="INV-42") == "saw:INV-42"


class TestParseJsonResponse:
    def test_parses_bare_json(self) -> None:
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_strips_markdown_code_fence(self) -> None:
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_recovers_object_embedded_in_prose(self) -> None:
        text = 'Sure! Here is the extraction:\n{"a": {"b": 2}}\nHope that helps.'
        assert parse_json_response(text) == {"a": {"b": 2}}

    def test_raises_when_no_object_present(self) -> None:
        with pytest.raises(ValueError, match="No JSON object found"):
            parse_json_response("just prose, no data")

    def test_raises_on_unterminated_object(self) -> None:
        with pytest.raises(ValueError, match="Unterminated JSON"):
            parse_json_response('prefix {"a": 1')
