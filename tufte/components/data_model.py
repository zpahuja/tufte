import base64
from typing import Dict, Optional, Union
from pydantic.dataclasses import dataclass


@dataclass
class Chart:
    spec: Optional[Union[str, Dict]] = None  # interactive specification e.g. vegalite
    status: Optional[bool] = None  # True if successful
    raster: Optional[str] = None
    code: Optional[str] = None  # code used to generate the visualization
    library: Optional[str] = None  # library used to generate the visualization
    error: Optional[Dict] = None  # error message if status is False

    def _repr_mimebundle_(self, include=None, exclude=None):
        bundle = {}
        if self.code:
            bundle["text/plain"] = self.code
        if self.raster:
            bundle["image/png"] = self.raster
        if self.spec:
            bundle["application/vnd.vegalite.v5+json"] = self.spec
        return bundle

    def savefig(self, path):
        if self.raster:
            with open(path, 'wb') as f:
                f.write(base64.b64decode(self.raster))
        else:
            raise FileNotFoundError("No raster image to save")
