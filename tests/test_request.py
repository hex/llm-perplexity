# ABOUTME: Tests for the pure request-building functions of the Perplexity plugin.
# ABOUTME: Runs without a network: history, citation stripping, images and request kwargs.
import base64
from types import SimpleNamespace

import llm
import pytest

from llm_perplexity import strip_citations

ONE_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415408d763f8cfc000000301010018dd8db00000000049454e44ae426082"
)


def make_prompt(text, system=None, **options):
    model = llm.get_model("sonar")
    return llm.Prompt(text, model=model, system=system, options=model.Options(**options))


def past_turn(question, answer, system=None):
    return SimpleNamespace(
        prompt=SimpleNamespace(prompt=question, system=system),
        text=lambda: answer,
    )


def test_strip_citations_removes_the_footer():
    text = "Paris.[1]\n\n## Citations:\n[1] Example - https://example.com\n"
    assert strip_citations(text) == "Paris.[1]"


def test_strip_citations_leaves_other_text_alone():
    assert strip_citations("Paris.") == "Paris."


def test_single_prompt_becomes_one_user_item():
    model = llm.get_model("sonar")
    assert model.build_input(make_prompt("Hello"), None) == [
        {"role": "user", "content": "Hello"}
    ]


def test_system_prompt_comes_first():
    model = llm.get_model("sonar")
    assert model.build_input(make_prompt("Hello", system="Be brief."), None) == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
    ]


def test_history_is_replayed_without_the_citations_footer():
    model = llm.get_model("sonar")
    conversation = SimpleNamespace(
        responses=[
            past_turn(
                "Capital of France?",
                "Paris.[1]\n\n## Citations:\n[1] Example - https://example.com\n",
            )
        ]
    )
    assert model.build_input(make_prompt("And Spain?"), conversation) == [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris.[1]"},
        {"role": "user", "content": "And Spain?"},
    ]


def test_system_prompt_from_the_first_turn_carries_into_follow_ups():
    model = llm.get_model("sonar")
    conversation = SimpleNamespace(
        responses=[past_turn("Capital of France?", "Paris.", system="Answer in French.")]
    )
    assert model.build_input(make_prompt("And Spain?"), conversation)[0] == {
        "role": "system",
        "content": "Answer in French.",
    }


def test_disabling_citations_adds_the_no_markers_instruction():
    model = llm.get_model("sonar")
    items = model.build_input(make_prompt("Hello", include_citations=False), None)
    assert items[0]["role"] == "system"
    assert "Do not include bracketed numeric citation markers" in items[0]["content"]


def test_image_path_becomes_an_input_image_part(tmp_path):
    image = tmp_path / "pixel.png"
    image.write_bytes(ONE_PIXEL_PNG)
    model = llm.get_model("sonar")
    items = model.build_input(make_prompt("What is this?", image_path=str(image)), None)
    assert items == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is this?"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,"
                    + base64.b64encode(ONE_PIXEL_PNG).decode("utf-8"),
                },
            ],
        }
    ]


def test_missing_image_raises_model_error(tmp_path):
    model = llm.get_model("sonar")
    with pytest.raises(llm.ModelError, match="Error processing image"):
        model.build_input(
            make_prompt("What is this?", image_path=str(tmp_path / "absent.png")), None
        )
