"""E2 full matrix with deepseek-chat via OpenAI-compatible API (local, no GPU).

Matches run_e2_pilot protocol exactly (same task selection, conditions,
CSV schema as run_e2_full_cloud.csv) but parallelized over tasks with
threads for API latency hiding.

Env (HTTPOpenAILLM): LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat  LLM_API_KEY=<key>

Run: python -m core_mas.run_e2_deepseek --out experiment_data/runs/run_e2_ds_full_cloud.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .allocators import STRATEGIES
from .datagen import generate
from .eval_agents import EvalAgents, HTTPOpenAILLM
from .eval_structural import AGENT_SET, make_budgets

STRATS = ["uniform_sharding", "frequency_replication",
          "task_aware_item", "core_mas"]
ORACLE_CAPS = {1.0}
VICTIMS = ["finance", "compliance", "logistics"]


def _row(seed, cap, sid, task, r, cond):
    row = {
        "seed": seed, "capacity": cap, "strategy": sid, "condition": cond,
        "task_id": task.task_id, "correct": r["correct"],
        "llm_correct": int(r["correct_llm"]),
        "structural_correct": int(r["correct_structural"]),
        "insufficient": int(r["insufficient_llm"]),
        "unsupported": int(r["unsupported_llm"]),
    }
    if "repair_answer" in r:
        row.update({
            "llm_correct_repair": int(r["correct_llm_repair"]),
            "repair_verified_n": r["repair_verified_n"],
            "repair_repaired_n": r["repair_repaired_n"],
            "repair_gate_pass": int(bool(r["repair_gate_pass"])),
            "insufficient_repair": int(r["insufficient_llm_repair"]),
            "raw_original_answer": r["coord_answer"],
            "raw_repair_answer": r["repair_answer"],
        })
    return row


def run_one(job, repair_flow=False):
    seed, cap, sid, task, victim, cond, llm, world, alloc = job
    ev = EvalAgents(world, alloc, llm, AGENT_SET)
    r = ev.run_task(task, victim=victim, repair_flow=repair_flow)
    return _row(seed, cap, sid, task, r, cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment_data/runs/run_e2_ds_full_cloud.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[200, 201, 202])
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--strategies", nargs="+", default=None,
                    help="strategy subset (default: all STRATS + oracle)")
    ap.add_argument("--repair-flow", action="store_true",
                    help="ECCR v2: verify -> gate -> targeted repair -> answer "
                         "(records both original and repair answers)")
    args = ap.parse_args()

    llm = HTTPOpenAILLM()
    print(f"llm -> {llm.base_url} model={llm.model}", flush=True)

    jobs = []
    t0 = time.time()
    for seed in args.seeds:
        world = generate(seed)
        for cap in (0.8, 1.0):
            budgets = make_budgets(world, cap)
            if args.strategies:
                sids = list(args.strategies)
            else:
                sids = list(STRATS)
                if cap in ORACLE_CAPS:
                    sids.append("oracle_witness")
            for sid in sids:
                alloc = STRATEGIES[sid](world.units, AGENT_SET, budgets,
                                        world.dev_tasks, world)
                for task in world.test_tasks[:args.tasks]:
                    jobs.append((seed, cap, sid, task, None, "s0", llm, world, alloc))
                    for v in VICTIMS:
                        jobs.append((seed, cap, sid, task, v, f"s1_{v}", llm, world, alloc))
    print(f"{len(jobs)} protocol executions, {args.workers} workers", flush=True)

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, j, repair_flow=args.repair_flow): j for j in jobs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:  # keep going; log failures
                j = futs[fut]
                print(f"FAIL {j[0]} {j[1]} {j[2]} {j[5]}: {e}", flush=True)
            done += 1
            if done % 500 == 0:
                el = time.time() - t0
                print(f"progress {done}/{len(jobs)} ({el:.0f}s, "
                      f"{done/el:.1f}/s, ETA {(len(jobs)-done)/(done/el)/60:.0f}min)",
                      flush=True)

    rows.sort(key=lambda r: (r["seed"], r["capacity"], r["strategy"],
                             r["condition"], r["task_id"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} rows -> {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
