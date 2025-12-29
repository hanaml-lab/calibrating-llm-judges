"""The criterion judge: one call per criterion, on evidence retrieved for it, read out as P(yes).

The client is any OpenAI-compatible endpoint (vLLM serves one) that returns top log-probabilities for the first
token. Nothing here is needed to run the estimator, which only consumes the probabilities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .evidence import Step

PROMPT = """You are checking one requirement of an agent's work.

Task: {task}

Requirement: {criterion}

Below are the steps of the run most relevant to it, each with its id.
{steps}

Decide from these steps alone whether the requirement is met. If the evidence is missing, answer no.
Answer yes or no, then list the step ids you relied on."""


@dataclass(frozen=True)
class Criterion:
    id: str
    text: str
    kind: str = "state"          # state, artifact, comms or process
    weight: float = 1.0
    locator: list[str] = field(default_factory=list)


def parse_verdict(top_logprobs: dict[str, float]) -> float:
    """P(yes) renormalised over the yes/no tokens among the top log-probabilities; 0.5 if neither appears."""
    yes = sum(math.exp(v) for k, v in top_logprobs.items() if k.strip().lower() == "yes")
    no = sum(math.exp(v) for k, v in top_logprobs.items() if k.strip().lower() == "no")
    if yes + no == 0:
        return 0.5
    return yes / (yes + no)


class CriterionJudge:
    def __init__(self, client, model: str, max_obs_chars: int = 6000):
        self.client = client
        self.model = model
        self.max_obs_chars = max_obs_chars

    def prompt(self, task: str, criterion: Criterion, evidence: list[Step]) -> str:
        steps = "\n".join(
            f"[step {s.index}] {s.action} {s.arguments}\n{s.observation[: self.max_obs_chars]}" for s in evidence
        )
        return PROMPT.format(task=task, criterion=criterion.text, steps=steps)

    def score(self, task: str, criterion: Criterion, evidence: list[Step]) -> float:
        """Raw judge score: P(yes) from the first generated token (temperature 0)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self.prompt(task, criterion, evidence)}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
        )
        top = response.choices[0].logprobs.content[0].top_logprobs
        return parse_verdict({t.token: t.logprob for t in top})
