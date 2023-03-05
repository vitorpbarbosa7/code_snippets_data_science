"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from boosting.pipelines import boosting as boost


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    pipelines_dict = {
        'boosting': boost.create_pipeline()
    }

    return pipelines_dict
