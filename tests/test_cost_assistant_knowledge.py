"""Cost assistant knowledge file + chatbot bundle."""

from ai_modules.cost_assistant_knowledge import load_cost_estimation_assistant_knowledge
from ai_modules.chatbot_module import ChatBotModule


def test_knowledge_markdown_loads():
    s = load_cost_estimation_assistant_knowledge()
    assert "AREA Covered" in s
    assert "Roof Area" in s
    assert "Grey structure" in s or "grey structure" in s.lower()


def test_full_bundle_has_policy_and_tables():
    b = ChatBotModule.get_cost_estimation_assistant_full_bundle()
    assert "policy" in b and "knowledge_tables" in b
    assert len(b["policy"]) > 500
    assert len(b["knowledge_tables"]) > 500
