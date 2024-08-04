MATPLOTLIB_CODE_TEMPLATE = f"""
import pandas as pd
import matplotlib.pyplot as plt
<imports>
# step by step plan for generating code -
def plot(data: pd.DataFrame):
    <stub> # only modify this section
    return plt

chart = plot(data) # Always include this line. No additional code beyond this line
""".strip()

MATPLOTLIB_INSTRUCTIONS = f"""
Do not include plt.show(). The plot method must return a matplotlib object (plt). Use BaseMap for charts that require a map.
""".strip()

SEABORN_CODE_TEMPLATE = f"""
import seaborn as sns
{MATPLOTLIB_CODE_TEMPLATE}
""".strip()

GGPLOT_CODE_TEMPLATE = """
import plotnine as p9
<imports>
# step by step plan for generating code -
def plot(data: pd.DataFrame):
    chart = <stub> # only modify this section
    return chart

chart = plot(data) # Always include this line. No additional code beyond this line
""".strip()

GGPLOT_INSTRUCTIONS = f"""
Do not include chart.show(). The plot method must return a ggplot object (chart). Use geom_map for charts that require a map.
""".strip()

ALTAIR_CODE_TEMPLATE = """
import altair as alt
<imports>
# step by step plan for generating code -
def plot(data: pd.DataFrame):
    <stub> # only modify this section
    return plt

chart = plot(data) # Always include this line. No additional code beyond this line
""".strip()

ALTAIR_INSTRUCTIONS = f"""
Ensure each field in the dataset is annotated with a type based on its semantic_type, such as :Q (quantitative), :O (ordinal), :N (nominal), :T (temporal), or :G (geographical). Use :T for fields where the semantic_type is either 'year' or 'date'. The plot function should construct and return an Altair chart object.
""".strip()

PLOTLY_CODE_TEMPLATE = """
import plotly.express as px
<imports>
# step by step plan for generating code -
def plot(data: pd.DataFrame):
    fig = <stub> # only modify this section
    return fig

chart = plot(data) # Always include this line. No additional code beyond this line
""".strip()

PLOTLY_INSTRUCTIONS = f"""
If calculating metrics (such as mean, median, mode) always use the option 'numeric_only=True' when applicable and available, avoid visualizations that require nbformat library. DO NOT inlcude fig.show()
""".strip()


class Scaffold:
    def get_template(self, library: str = "altair"):
        if library == "matplotlib":
            return MATPLOTLIB_CODE_TEMPLATE, MATPLOTLIB_INSTRUCTIONS
        elif library == "seaborn":
            return SEABORN_CODE_TEMPLATE, MATPLOTLIB_INSTRUCTIONS
        elif library == "ggplot":
            return GGPLOT_CODE_TEMPLATE, GGPLOT_INSTRUCTIONS
        elif library == "altair":
            return ALTAIR_CODE_TEMPLATE, ALTAIR_INSTRUCTIONS
        elif library == "plotly":
            return PLOTLY_CODE_TEMPLATE, PLOTLY_INSTRUCTIONS
        else:
            raise ValueError("Unsupported library. Choose from 'matplotlib', 'seaborn', 'ggplot', 'altair', or 'plotly'.")
