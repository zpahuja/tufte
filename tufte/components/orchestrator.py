import logging
import pandas as pd
from typing import Dict, List, Union

from .code_executor import CodeExecutor
from .data_model import Chart
from .goal_explorer import GoalExplorer
from .summarizer import Summarizer
from .utils import read_dataframe
from .viz_generator import VizGenerator

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class Orchestrator:
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL) -> None:
        self.data = None
        self.oai_model = model
        self.summarizer = Summarizer(model=self.oai_model)
        self.goal_explorer = GoalExplorer(model=self.oai_model)
        self.viz_generator = VizGenerator(model=self.oai_model)
        self.code_executor = CodeExecutor()

    def summarize(self, data: Union[pd.DataFrame, str], n_samples: int = 3, enrich=False) -> Dict:
        if isinstance(data, str):
            self.data = read_dataframe(data)
        return self.summarizer.summarize(data=self.data, n_samples=n_samples, enrich=enrich)

    def explore_goals(self, summary: Dict, n_goals: int = 5) -> List[Dict]:
        return self.goal_explorer.generate_goals(summary=summary, n_goals=n_goals)

    def visualize(self, summary: Dict, goal: Dict, library: str = "altair", debug=False) -> List:
        if isinstance(goal, str):
            goal = {"question": goal, "visualization": goal, "rationale": ""}
        code = self.viz_generator.generate_code(summary=summary, goal=goal, library=library)
        charts = self.code_executor.execute_code(code, data=self.data, library=library, return_error=debug)
        return [Chart(**chart) for chart in charts]