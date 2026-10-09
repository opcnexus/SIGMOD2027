from .m01_event_extraction import M01EventExtraction
from .m02_causal_micro import M02CausalMicro
from .m03_eventweave import M03EventWeave
from .m04_blackbox_forensics import M04BlackBoxForensics
from .m05_reflexion import M05Reflexion
from .m06_expel import M06ExpeL
from .m07_lats import M07LATS
from .m08_ananke import M08Ananke
from .m09_trace2attck import M09Trace2ATTACK
from .m10_cascade_inference import M10CascadeInference

ALL = [
    M01EventExtraction,
    M02CausalMicro,
    M03EventWeave,
    M04BlackBoxForensics,
    M05Reflexion,
    M06ExpeL,
    M07LATS,
    M08Ananke,
    M09Trace2ATTACK,
    M10CascadeInference,
]

BY_NAME = {m.name: m for m in ALL}


def build(name):
    return BY_NAME[name]()
