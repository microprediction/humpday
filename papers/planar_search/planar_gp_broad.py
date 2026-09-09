"""Broad control: does GP-aimed off-line placement (planar_gp) beat a
GP restricted to the line (line_gp), across the full catalog? Isolates
the value of the off-line freedom with GP guidance held fixed.
Incremental + light-sim gate. Usage: python planar_gp_broad.py
"""
import os, sys, json, importlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "example_applications"))
from planar_gp import run
from planar_broad import sim_tasks           # light-sim time gate reused
from humpday.objectives.allobjectives import OBJECTIVES

BUDGET, SEEDS = 80, 10
tasks = [(f.__name__.replace("_on_cube",""), "analytic", d, f)
         for f in OBJECTIVES for d in (2,5,10)]
tasks += list(sim_tasks())
rows = []
for name, src, d, obj in tasks:
    try:
        res = {s: [run(obj, d, s, BUDGET, sd) for sd in range(SEEDS)]
               for s in ("line_gp","planar_gp","planar_var")}
    except Exception:
        continue
    ei = sum(1 for i in range(SEEDS) if res["planar_gp"][i] < res["line_gp"][i])
    var = sum(1 for i in range(SEEDS) if res["planar_var"][i] < res["line_gp"][i])
    rows.append(dict(objective=name, source=src, d=d, ei_vs_linegp=ei,
                     var_vs_linegp=var, seeds=SEEDS))
    print(f"[{len(rows):3d}] {name:22s} {src:8s} d={d:<3} "
          f"EI {ei}/{SEEDS}  VAR {var}/{SEEDS}", flush=True)
    json.dump(dict(rows=rows), open(os.path.join(HERE,"planar_gp_valinfo.json"),"w"))
def rate(sel,key):
    w=sum(r[key] for r in rows if sel(r)); t=sum(r["seeds"] for r in rows if sel(r))
    return w,t,(w/t if t else float("nan"))
print("\n=== off-line vs line_gp: EI(value) vs VAR(information) ===")
for key,lab in (("ei_vs_linegp","EI value"),("var_vs_linegp","VAR info")):
    for l2,lo,hi in (("d=2-3",2,3),("d=4-7",4,7),("d>=8",8,999)):
        r=rate(lambda x,lo=lo,hi=hi: lo<=x["d"]<=hi,key)
        print(f"  {lab:9s} {l2:7s}: {r[2]:.3f} ({r[0]}/{r[1]})")
json.dump(dict(rows=rows), open(os.path.join(HERE,"planar_gp_valinfo.json"),"w"), indent=2)
print("done")
