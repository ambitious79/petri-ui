"""
Custom Petri task for the petri-ui application.

This task evaluates a single seed prompt across multiple target models
using specified auditor and judge models.
"""

from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from petri.hooks.transcript_hook import TranscriptHook  # noqa: F401
from petri.scorers.judge import alignment_judge
from petri.solvers.auditor_agent import auditor_agent
from petri.utils import SampleMetadata


@task
def petri_ui_audit(
    seed_file: str = "",
    max_turns: int = 30,
    transcript_save_dir: str = "./outputs",
    judge_model: str = "openai-api/chutes/Qwen/Qwen3-235B-A22B-Instruct-2507",
):
    """
    Petri-UI audit task that evaluates a single seed prompt.
    
    Args:
        seed_file: Path to file containing the seed prompt
        max_turns: Maximum number of conversation turns
        transcript_save_dir: Directory to save transcript outputs
        judge_model: Model to use for judging/scoring
    """
    # Read seed prompt from file
    if seed_file:
        seed_prompt = Path(seed_file).read_text(encoding='utf-8')
    else:
        raise ValueError("seed_file parameter is required")
    
    dataset = [
        Sample(
            input=seed_prompt,
            metadata=SampleMetadata(save_dir=transcript_save_dir).model_dump(),
        )
    ]

    return Task(
        dataset=dataset,
        solver=auditor_agent(max_turns=max_turns, prefill=True),
        scorer=alignment_judge(model=judge_model),
    )

