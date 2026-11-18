# -*- coding: utf-8 -*-
"""第五轮意见关键实验（纯 CPU，无 LLM 调用）。

1) 2x2 权限 x 协调器存储对照（3 种子 x 4 容量 x 6 策略）：
   A 现行权限 / 协调器可存（= 主协议）
   B 放宽权限（任一领域智能体可持有任一单元）/ 协调器可存
   C 现行权限 / 协调器不可存（预算转给 3 领域智能体，公平口径）
   D 放宽权限 / 协调器不可存  <- 审稿人要求的决定性对照
2) bundle 事实数上限 4：主基准 M=8 全容量；规模反转格 (M,N) in {(64,3),(128,3),(128,6)}
3) 主基准按任务所需事实数分层覆盖率（80% 容量）
4) uniform_sharding vs independent_summary 分配诊断

输出: experiment_data/runs/run_e1_round5/
"""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

from core_mas.allocators import STRATEGIES, Budgets, expected_coverage
from core_mas.datagen import generate
from core_mas.eval_structural import AGENT_SET, make_budgets
from core_mas.eval_scale import make_budgets_n
from core_mas.failure_scenarios import FailureSpec

DOMAIN = [a for a in AGENT_SET if a != "coordinator"]
SEEDS = [200, 201, 202]
CAPS = [0.4, 0.6, 0.8, 1.0]
STRATS = ["core_mas", "frequency_replication", "task_aware_item",
          "traditional_placement", "uniform_sharding", "independent_summary"]
OUT = Path(__file__).resolve().parents[1] / "experiment_data" / "runs" / "run_e1_round5"


def eval_fw(world, alloc, agents):
    rng = random.Random(42)
    s0 = expected_coverage(world.test_tasks, alloc, FailureSpec("s0"), agents, rng)
    s1 = expected_coverage(world.test_tasks, alloc, FailureSpec("s1"), agents, rng)
    s2 = expected_coverage(world.test_tasks, alloc, FailureSpec("s2"), agents, rng, s2_samples=32)
    return s0, s1, s2, (s1 + s2) / 2


def widen(world):
    w = copy.deepcopy(world)
    for u in w.units:
        u.access_scope = list(DOMAIN)
    return w


def arm_rows(arm, world, budgets, agents, kw=None):
    rows = []
    for sid in STRATS:
        t0 = time.time()
        fn_kw = dict(kw) if kw else {}
        alloc = STRATEGIES[sid](world.units, agents, budgets,
                               world.dev_tasks, world, **fn_kw)
        s0, s1, s2, fw = eval_fw(world, alloc, agents)
        rows.append({"arm": arm, "seed": world.seed, "capacity": world._cap,
                     "strategy": sid, "cov_s0": round(s0, 4), "cov_s1": round(s1, 4),
                     "cov_s2": round(s2, 4), "fault_weighted": round(fw, 4),
                     "seconds": round(time.time() - t0, 2)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"],
                    help="A/B/C/D 2x2 arms")
    ap.add_argument("--cap4", action="store_true", help="bundle cap 4 runs")
    ap.add_argument("--scale-cap4", action="store_true", help="M=128/64 cap-4 cells")
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1) 2x2 arms ----
    if args.arms:
        rows = []
        for seed in SEEDS:
            base = generate(seed)
            for cap in CAPS:
                total = int(base.baseline_cost * cap)
                per4 = max(1, total // len(AGENT_SET))
                per3 = max(1, total // len(DOMAIN))
                store = Budgets(per_agent={a: per4 for a in AGENT_SET}, total=total)
                nostore = Budgets(per_agent={**{a: per3 for a in DOMAIN},
                                             "coordinator": 0}, total=total)
                worlds = {"current": base, "relaxed": widen(base)}
                for arm, (perm, bud, ags) in {
                    "A": ("current", store, AGENT_SET),
                    "B": ("relaxed", store, AGENT_SET),
                    # coordinator truly absent: only the 3 domain agents store,
                    # and s1 victimizes exactly those 3 (no degenerate no-op victim)
                    "C": ("current", nostore, DOMAIN),
                    "D": ("relaxed", nostore, DOMAIN),
                }.items():
                    if arm not in args.arms:
                        continue
                    w = worlds[perm]
                    w._cap = cap
                    rows.extend(arm_rows(arm, w, bud, ags))
                    print(f"arm {arm} seed={seed} cap={cap} done", flush=True)
        with open(OUT / "e1_round5_arms.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("saved e1_round5_arms.csv", len(rows), "rows")

    # ---- 2a) bundle cap 4 on main benchmark ----
    if args.cap4:
        rows = []
        for seed in SEEDS:
            world = generate(seed)
            for cap in CAPS:
                budgets = make_budgets(world, cap)
                t0 = time.time()
                alloc = STRATEGIES["core_mas"](world.units, AGENT_SET, budgets,
                                               world.dev_tasks, world,
                                               bundle_max_facts=4)
                s0, s1, s2, fw = eval_fw(world, alloc, AGENT_SET)
                rows.append({"seed": seed, "capacity": cap, "strategy": "core_mas_cap4",
                             "cov_s0": round(s0, 4), "cov_s1": round(s1, 4),
                             "cov_s2": round(s2, 4), "fault_weighted": round(fw, 4),
                             "seconds": round(time.time() - t0, 2)})
                print(f"cap4 main seed={seed} cap={cap} fw={fw:.4f}", flush=True)
        with open(OUT / "e1_round5_cap4_main.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("saved e1_round5_cap4_main.csv")

    # ---- 2b) bundle cap 4 at scale inversion cells ----
    if args.scale_cap4:
        rows = []
        for (m, n) in [(64, 3), (128, 3), (128, 6)]:
            world = generate(200, n_agents=n, n_suppliers=m)
            agents = [a for a in world.units[0].access_scope] + ["coordinator"]
            agents = sorted(set(a for u in world.units for a in u.access_scope)) + ["coordinator"]
            budgets = make_budgets_n(world, 0.8, agents)
            for sid, kw in [("core_mas", {}), ("core_mas_cap4", {"bundle_max_facts": 4}),
                            ("frequency_replication", {})]:
                t0 = time.time()
                alloc = STRATEGIES[sid.split("_cap4")[0]](world.units, agents, budgets,
                                                          world.dev_tasks, world, **kw)
                s0, s1, s2, fw = eval_fw(world, alloc, agents)
                rows.append({"M": m, "N": n, "strategy": sid,
                             "cov_s0": round(s0, 4), "cov_s1": round(s1, 4),
                             "cov_s2": round(s2, 4), "fault_weighted": round(fw, 4),
                             "seconds": round(time.time() - t0, 2)})
                print(f"scale-cap4 M={m} N={n} {sid} fw={fw:.4f}", flush=True)
        with open(OUT / "e1_round5_scale_cap4.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("saved e1_round5_scale_cap4.csv")

    # ---- 3) fact-count stratification on main benchmark ----
    if args.stratify:
        rows = []
        for seed in SEEDS:
            world = generate(seed)
            budgets = make_budgets(world, 0.8)
            groups = {}
            for t in world.test_tasks:
                need = max(len(b) for b in t.witness_bundles)
                groups.setdefault(min(need, 4), []).append(t)
            for sid in ["core_mas", "frequency_replication", "traditional_placement"]:
                alloc = STRATEGIES[sid](world.units, AGENT_SET, budgets,
                                        world.dev_tasks, world)
                for g, tasks in sorted(groups.items()):
                    rng = random.Random(42)
                    s1 = expected_coverage(tasks, alloc, FailureSpec("s1"), AGENT_SET, rng)
                    s2 = expected_coverage(tasks, alloc, FailureSpec("s2"), AGENT_SET, rng,
                                           s2_samples=32)
                    rows.append({"seed": seed, "strategy": sid, "fact_group": g,
                                 "n_tasks": len(tasks),
                                 "fw": round((s1 + s2) / 2, 4)})
        with open(OUT / "e1_round5_stratify.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("saved e1_round5_stratify.csv")

    # ---- 4) uniform vs independent_summary diagnosis ----
    if args.diag:
        world = generate(200)
        budgets = make_budgets(world, 0.8)
        au = STRATEGIES["uniform_sharding"](world.units, AGENT_SET, budgets,
                                            world.dev_tasks, world)
        ai = STRATEGIES["independent_summary"](world.units, AGENT_SET, budgets,
                                               world.dev_tasks, world)
        def sig(a):
            return sorted((u.unit_id for u in v), key=str) and \
                {k: sorted(u.unit_id for u in v) for k, v in a.items()}
        print("uniform == indep:", sig(au) == sig(ai))
        print("uniform:", {k: [u.unit_id for u in v] for k, v in au.items()})
        print("indep:  ", {k: [u.unit_id for u in v] for k, v in ai.items()})
        from collections import Counter
        kc = Counter((u.fact_id, u.kind) for u in world.units)
        per_fact = {}
        for u in world.units:
            per_fact.setdefault(u.fact_id, []).append((u.kind, u.token_cost))
        print("facts:", len(world.facts), "units:", len(world.units))
        print("example facts:", dict(list(per_fact.items())[:6]))
        # baseline_cost & budget
        print("baseline_cost:", world.baseline_cost, "| budget@0.8:", budgets.total,
              "per:", {a: budgets.per_agent[a] for a in AGENT_SET})


if __name__ == "__main__":
    main()
