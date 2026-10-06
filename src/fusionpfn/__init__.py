"""FusionPFN: pretrained, in-context fusion of model committees."""
from .fuse import fusionpfn_predict as fuse
from .fuse import load_model
from .model import FusionPFN

__all__ = ["FusionPFN", "fuse", "load_model"]
