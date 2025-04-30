import sys
from pathlib import Path
import pandas as pd
import numpy as np

print((Path(__file__).parent.parent / "dsl4ras-pfd").resolve().__str__())
sys.path.append((Path(__file__).parent.parent / "dsl4ras-pfd").resolve().__str__())
from SCARA_bottom_up_mappings import ScaraSystem


sys = ScaraSystem()
dv = pd.read_csv("./problems/DSL4RAS/input/dv_space.csv")
qoi = pd.read_csv("./problems/DSL4RAS/input/qoi_space.csv")

seed = 42
rng = np.random.default_rng(seed)

dv_samples = rng.uniform(
    low=dv["Lower"].values,
    high=dv["Upper"].values,
    size=(1, dv.shape[0]),
)

qoi_results = sys.evaluate_qois(dv=dv_samples[0])
print(qoi_results)
