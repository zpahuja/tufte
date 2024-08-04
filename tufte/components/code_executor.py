import ast
import base64
import importlib
import io
import logging
import traceback
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio


logger = logging.getLogger(__name__)


class CodeExecutor:
    def get_globals_dict(self, code_string: str, data: pd.DataFrame):
        tree = ast.parse(code_string)
        imported_modules = [
            (alias.name, alias.asname, importlib.import_module(alias.name))
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            (
                f"{node.module}.{alias.name}",
                alias.asname,
                getattr(importlib.import_module(node.module), alias.name),
            )
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        ]

        globals_dict = {
            alias or module_name.split(".")[-1]: obj
            for module_name, alias, obj in imported_modules
        }
        globals_dict.update({"pd": pd, "data": data, "plt": plt})
        return globals_dict

    def execute_code(
        self,
        code_specs: List[str],
        data: Any,
        library="altair",
        return_error: bool = False,
    ) -> Any:
        library_handlers = {
            "altair": self._handle_altair,
            "matplotlib": self._handle_matplotlib,
            "seaborn": self._handle_matplotlib,  # seaborn uses matplotlib's handler
            "ggplot": self._handle_ggplot,
            "plotly": self._handle_plotly,
        }

        if library not in library_handlers:
            raise Exception(
                f"Unsupported library. Supported libraries are altair, matplotlib, seaborn, ggplot, plotly. You provided {library}"
            )

        return library_handlers[library](code_specs, data, return_error)

    def _handle_altair(self, code_specs: List[str], data: Any, return_error: bool):
        results = []
        for code in code_specs:
            try:
                ex_locals = self.get_globals_dict(code, data)
                exec(code, ex_locals)
                chart = ex_locals["chart"]
                vega_spec = chart.to_dict()
                del vega_spec["data"]
                vega_spec.pop("datasets", None)

                results.append(
                    {
                        "spec": vega_spec,
                        "status": True,
                        "code": code,
                        "library": "altair",
                    }
                )
            except Exception as exception_error:
                logger.error(f"{code} ****\n{str(exception_error)}")
                logger.error(traceback.format_exc())
                if return_error:
                    results.append(
                        {
                            "status": False,
                            "code": code,
                            "library": "altair",
                            "error": {
                                "message": str(exception_error),
                                "traceback": traceback.format_exc(),
                            },
                        }
                    )
        return results

    def _handle_matplotlib(self, code_specs: List[str], data: Any, return_error: bool):
        results = []
        for code in code_specs:
            try:
                ex_locals = self.get_globals_dict(code, data)
                exec(code, ex_locals)
                plt = ex_locals["chart"]
                if plt:
                    buf = io.BytesIO()
                    # plt.box(False)
                    # plt.grid(color="lightgray", linestyle="dashed", zorder=-10)
                    plt.savefig(buf, format="png", dpi=100, pad_inches=0.2)
                    buf.seek(0)
                    plot_data = base64.b64encode(buf.read()).decode("ascii")
                    plt.close()
                results.append(
                    {
                        "status": True,
                        "raster": plot_data,
                        "code": code,
                        "library": "matplotlib",
                    }
                )
            except Exception as exception_error:
                logger.error(f"{code_specs[0]} ****\n{str(exception_error)}")
                logger.error(traceback.format_exc())
                if return_error:
                    results.append(
                        {
                            "status": False,
                            "code": code,
                            "library": "matplotlib",
                            "error": {
                                "message": str(exception_error),
                                "traceback": traceback.format_exc(),
                            },
                        }
                    )
        return results

    def _handle_ggplot(self, code_specs: List[str], data: Any, return_error: bool):
        results = []
        for code in code_specs:
            try:
                ex_locals = self.get_globals_dict(code, data)
                exec(code, ex_locals)
                chart = ex_locals["chart"]
                if plt:
                    buf = io.BytesIO()
                    chart.save(buf, format="png")
                    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                results.append(
                    {
                        "status": True,
                        "raster": plot_data,
                        "code": code,
                        "library": "ggplot",
                    }
                )
            except Exception as exception_error:
                logger.error(f"{code} {traceback.format_exc()}")
                if return_error:
                    results.append(
                        {
                            "status": False,
                            "code": code,
                            "library": "ggplot",
                            "error": {
                                "message": str(exception_error),
                                "traceback": traceback.format_exc(),
                            },
                        }
                    )
        return results

    def _handle_plotly(self, code_specs: List[str], data: Any, return_error: bool):
        results = []
        for code in code_specs:
            try:
                ex_locals = self.get_globals_dict(code, data)
                exec(code, ex_locals)
                chart = ex_locals["chart"]

                if pio:
                    chart_bytes = pio.to_image(chart, "png")
                    plot_data = base64.b64encode(chart_bytes).decode("utf-8")

                    results.append(
                        {
                            "status": True,
                            "raster": plot_data,
                            "code": code,
                            "library": "plotly",
                        }
                    )
            except Exception as exception_error:
                logger.error(f"{code} {traceback.format_exc()}")
                if return_error:
                    results.append(
                        {
                            "status": False,
                            "code": code,
                            "library": "plotly",
                            "error": {
                                "message": str(exception_error),
                                "traceback": traceback.format_exc(),
                            },
                        }
                    )
        return results
