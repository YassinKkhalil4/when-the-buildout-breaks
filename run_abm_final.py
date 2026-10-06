"""Final Model F runs for the working paper: upgrade ablation (all nine upgrades), cliff curve, single-point-of-failure knockouts.
Usage: python run_abm_final.py [seeds] [all|ablation|cliff|knockout]   (about 14 minutes on 2 cores for 'all'; writes abm_final_*.json)"""
import json, sys, time
import ai_bust_abm as m

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    only = sys.argv[2] if len(sys.argv) > 2 else "all"
    t0 = time.time()
    if only in ("all", "ablation"):
        raw = m.upgrade_ablation(shocks=(0.10, 0.20, 0.30), n_seeds=n)
        json.dump({"n_seeds": raw["n_seeds"], "shocks": raw["shocks"], "table": m.ablation_table(raw), "rows": raw["rows"]},
                  open("abm_final_ablation.json", "w"), default=float)
        print("ablation done", round(time.time() - t0), flush=True)
    if only in ("all", "cliff"):
        json.dump(m.cliff_curve(n_seeds=32), open("abm_final_cliff.json", "w"), default=float)
        print("cliff done", round(time.time() - t0), flush=True)
    if only in ("all", "knockout"):
        ko = {mode: m.knockout_ranking(shock=0.15, n_seeds=24, mode=mode) for mode in ("bipartite", "archetype")}
        json.dump(ko, open("abm_final_knockout.json", "w"), default=float)
        print("knockout done", round(time.time() - t0), flush=True)
