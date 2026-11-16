"""Run the GA and write the schedule, timetables, fitness plot and summary to outputs/."""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fitness import SOFT_WEIGHTS, W_HARD, fitness, sessions
from ga import GAConfig, run
from instance import (DAY_NAMES, N_PROFS, N_ROOMS, N_SLOTS, course_name, make_instance, prof_name,
                      room_name, section_name)

SIZE_NAMES = ["Small", "Average", "Large"]


def write_schedule(best, inst, out):
    prof, day, slot, room = sessions(best)
    rows = []
    for k in range(len(prof)):
        s = k // 2
        rows.append({
            "section": section_name(s), "course": course_name(inst.sec_course[s]),
            "session": k % 2 + 1, "professor": prof_name(prof[k]), "day": DAY_NAMES[day[k]],
            "timeslot": slot[k] + 1, "room": room_name(room[k]),
            "room_size": SIZE_NAMES[inst.room_size[room[k]]],
            "registration_size": SIZE_NAMES[inst.sec_size[s]],
            "favorite": bool(inst.favorite[prof[k], inst.sec_course[s]]),
        })
    rows.sort(key=lambda r: (DAY_NAMES.index(r["day"]), r["timeslot"], r["room"]))
    with open(out / "schedule.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return prof, day, slot, room


def grid(title, cells):
    """cells: dict (day, slot) -> text. Returns a fixed-width weekly grid."""
    width = 13
    lines = [title, "slot | " + " | ".join(d.ljust(width) for d in DAY_NAMES)]
    for t in range(N_SLOTS):
        row = [cells.get((d, t), "").ljust(width) for d in range(len(DAY_NAMES))]
        lines.append(f"  {t + 1}  | " + " | ".join(row))
        if t == 2:
            lines.append("-" * len(lines[1]))  # morning / afternoon divider
    return "\n".join(lines) + "\n"


def write_timetables(prof, day, slot, room, out, inst):
    by_room = [dict() for _ in range(N_ROOMS)]
    by_prof = [dict() for _ in range(N_PROFS)]
    load = np.bincount(prof, minlength=N_PROFS)

    def put(cells, key, text):  # a clash shows both entries instead of hiding one of them
        cells[key] = f"{cells[key]} / {text}" if key in cells else text

    for k in range(len(prof)):
        sec = section_name(k // 2)
        put(by_room[room[k]], (day[k], slot[k]), f"{sec} {prof_name(prof[k])}")
        put(by_prof[prof[k]], (day[k], slot[k]), f"{sec} {room_name(room[k])}")
    with open(out / "timetable_rooms.txt", "w") as f:
        for r in range(N_ROOMS):
            f.write(grid(f"\n{room_name(r)} ({SIZE_NAMES[inst.room_size[r]]})", by_room[r]))
    with open(out / "timetable_profs.txt", "w") as f:
        for p in range(N_PROFS):
            favs = ", ".join(course_name(c) for c in np.flatnonzero(inst.favorite[p]))
            can = ", ".join(course_name(c) for c in np.flatnonzero(inst.can_teach[p]))
            f.write(grid(f"\n{prof_name(p)}  load={load[p]}  can teach: {can}  "
                         f"favourites: {favs}", by_prof[p]))


def plot(hist, out):
    g = np.arange(len(hist.best))
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    a.plot(g, hist.best, label="best penalty")
    a.plot(g, hist.mean, label="mean penalty", alpha=0.6)
    a.set_yscale("log")
    a.set_ylabel(f"penalty = {W_HARD}·H + S\n(fitness = 1 / (1 + penalty))")
    a.set_title("GA convergence")
    a.legend()
    a.grid(True, which="both", alpha=0.3)
    b.plot(g, hist.hard, label="hard violations H (best)", color="tab:red")
    b.plot(g, hist.soft, label="soft penalty S (best)", color="tab:green")
    b.set_yscale("symlog", linthresh=1)
    b.set_xlabel("generation")
    b.set_ylabel("count")
    b.legend()
    b.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fitness.png", dpi=150)
    fig.savefig(out / "fitness.pdf")  # vector copy for the report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="seed for both instance and GA")
    ap.add_argument("--generations", type=int, default=GAConfig.generations)
    ap.add_argument("--pop", type=int, default=GAConfig.pop_size)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    inst = make_instance(args.seed)
    cfg = GAConfig(pop_size=args.pop, generations=args.generations, seed=args.seed)
    t0 = time.time()
    best, (penalty, terms), hist = run(inst, cfg)
    runtime = time.time() - t0

    prof, day, slot, room = write_schedule(best, inst, out)
    write_timetables(prof, day, slot, room, out, inst)
    plot(hist, out)
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_penalty", "mean_penalty", "best_H", "best_S"])
        w.writerows(zip(range(len(hist.best)), hist.best, hist.mean, hist.hard, hist.soft))
    summary = {
        "seed": args.seed, "generations_run": len(hist.best) - 1, "runtime_s": round(runtime, 1),
        "config": vars(cfg), "penalty": int(penalty), "fitness": fitness(penalty),
        "terms": terms, "soft_weights": SOFT_WEIGHTS,
        "first_feasible_generation": next((g for g, h in enumerate(hist.hard) if h == 0), None),
        "initial_best": {"penalty": hist.best[0], "H": hist.hard[0], "S": hist.soft[0]},
        "instance": {
            "rooms_per_size": np.bincount(inst.room_size, minlength=3).tolist(),
            "sections_per_size": np.bincount(inst.sec_size, minlength=3).tolist(),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    write_numbers(summary, inst, prof, room, out)


def write_numbers(summary, inst, prof, room, out):
    """LaTeX macros so the report quotes the numbers of the latest run instead of typed copies."""
    n = len(prof)
    size = np.repeat(inst.sec_size, 2)
    course = np.repeat(inst.sec_course, 2)
    t = summary["terms"]
    values = {
        "Seed": summary["seed"],
        "GenerationsRun": summary["generations_run"],
        "RuntimeMin": round(summary["runtime_s"] / 60),
        "FirstFeasible": summary["first_feasible_generation"],
        "InitPenalty": f"{summary['initial_best']['penalty']:,}".replace(",", "\\,"),
        "InitH": summary["initial_best"]["H"],
        "InitS": summary["initial_best"]["S"],
        "FinalPenalty": summary["penalty"],
        "FinalFitness": f"{summary['fitness'] * 1000:.1f}",  # times 10^-3
        "ReductionFactor": int(round(summary["initial_best"]["penalty"] / max(summary["penalty"], 1), -1)),
        "GapCount": t["prof_gap"], "MixedDays": t["prof_mixed_day"],
        "SplitCount": t["course_split"], "WasteCount": t["room_waste"],
        "NoFavCount": t["no_favorite"],
        "PreferredHalfPct": f"{100 * (1 - t['course_split'] / n):.1f}",
        "ExactRoomPct": f"{100 * np.mean(inst.room_size[room] == size):.1f}",
        "FavSessionPct": f"{100 * np.mean(inst.favorite[prof, course]):.1f}",
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()]
    (out / "numbers.tex").write_text("% generated by main.py\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
