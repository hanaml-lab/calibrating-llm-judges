import math

from parc.evidence import Step, retrieve
from parc.judge import CriterionJudge, Criterion, parse_verdict


STEPS = [
    Step(0, "read_file", "inbox/ticket-4411.txt", "Customer asks for a refund of order 8812."),
    Step(1, "query_db", "orders where id=8812", "status=shipped amount=129.00"),
    Step(2, "run", "ls -la", "drwxr-xr-x reports"),
    Step(3, "call_api", "refunds.create order=8812 amount=129.00", "refund_id=R-77 status=pending"),
    Step(4, "send_message", "to=customer", "Your refund R-77 for order 8812 has been issued."),
]


def test_a_step_that_touches_the_locator_outranks_text_matches():
    hits = retrieve(STEPS, "A refund is created for the full order amount", locator=["refunds.create"], k=1)
    assert [h.index for h in hits] == [3]


def test_retrieval_keeps_the_runs_order_and_respects_k():
    hits = retrieve(STEPS, "refund order 8812", locator=[], k=3)
    assert len(hits) == 3
    assert [h.index for h in hits] == sorted(h.index for h in hits)


def test_communication_criteria_always_see_the_final_message():
    hits = retrieve(STEPS, "The customer is told the refund id", locator=[], k=1, kind="comms")
    assert STEPS[-1].index in [h.index for h in hits]


def test_prompt_shows_the_criterion_and_the_step_ids():
    judge = CriterionJudge(client=None, model="Qwen3-32B")
    crit = Criterion("c1", "A refund is created for the full order amount", kind="state", locator=["refunds.create"])
    prompt = judge.prompt("Handle ticket 4411.", crit, retrieve(STEPS, crit.text, crit.locator, k=2))
    assert "A refund is created for the full order amount" in prompt
    assert "[step 3]" in prompt


def test_verdict_probability_comes_from_the_yes_and_no_logprobs():
    p = parse_verdict({"yes": math.log(0.3), "no": math.log(0.1), "maybe": math.log(0.6)})
    assert abs(p - 0.75) < 1e-12
    assert parse_verdict({" Yes": 0.0}) == 1.0
    assert parse_verdict({}) == 0.5
