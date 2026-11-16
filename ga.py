"""Genetic algorithm for the class-scheduling problem."""
import time
from dataclasses import dataclass, field

import numpy as np

from fitness import evaluate
from instance import (DAY_PAIRS, MAX_LOAD, MORNING, N_COURSES, N_PROFS, N_ROOMS, N_SECTIONS,
                      N_SLOTS, N_TIMES, SEC_PER_COURSE)



@dataclass
class GAConfig:
    pop_size: int = 150
    generations: int = 5000
    tournament: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.0005  # per gene
    repairs: int = 3  # conflicting sections repaired per child
    time_tries: int = 12  # candidate times sampled when repairing a section
    elite: int = 2
    patience: int = 400  # stop after this many generations without improvement (once feasible)
    seed: int = 0


@dataclass
class History:
    best: list = field(default_factory=list)
    mean: list = field(default_factory=list)
    hard: list = field(default_factory=list)
    soft: list = field(default_factory=list)


def random_population(inst, rng, n):
    """Constraint-aware random individuals: qualified professor, large-enough rooms."""
    qualified = np.stack(inst.qualified)  # (courses, profs per course); equal length by construction
    pad = np.full((3, len(inst.room_size)), -1)
    lens = np.array([len(r) for r in inst.rooms_for_size])
    for s, rooms in enumerate(inst.rooms_for_size):
        pad[s, :len(rooms)] = rooms

    pop = np.empty((n, N_SECTIONS, 6), np.int64)
    pop[:, :, 0] = qualified[inst.sec_course, rng.integers(0, qualified.shape[1], (n, N_SECTIONS))]
    pop[:, :, 1] = rng.integers(0, len(DAY_PAIRS), (n, N_SECTIONS))
    pop[:, :, 2:4] = rng.integers(0, N_SLOTS, (n, N_SECTIONS, 2))
    pick = (rng.random((n, N_SECTIONS, 2)) * lens[inst.sec_size][None, :, None]).astype(int)
    pop[:, :, 4:6] = pad[inst.sec_size[None, :, None], pick]
    return pop


def repair(kid, inst, rng, cfg):
    """Conflict-directed repair (in place). Moves a few sections to a time where the professor is
    free, inside the course's majority half-day (soft S3), using the smallest free rooms that fit
    (least waste, S4). Targets sections in a hard conflict; when the child has none, it targets
    sections that sit in their course's minority half. A move is only accepted if it creates no
    clash."""
    t = DAY_PAIRS[kid[:, 1]] * N_SLOTS + kid[:, 2:4]  # (sections, 2) time cell of each session
    profs = np.repeat(kid[:, :1], 2, 1)
    room_occ = np.zeros((N_ROOMS, N_TIMES), np.int32)
    prof_occ = np.zeros((N_PROFS, N_TIMES), np.int32)
    np.add.at(room_occ, (kid[:, 4:6], t), 1)
    np.add.at(prof_occ, (profs, t), 1)
    load = 2 * np.bincount(kid[:, 0], minlength=N_PROFS)
    unfit = ~inst.can_teach[kid[:, 0], inst.sec_course]

    # Hard conflicts of this child, computed from its own occupancy (crossover can create new ones).
    clash = (room_occ[kid[:, 4:6], t] > 1) | (prof_occ[profs, t] > 1)
    clash |= inst.room_size[kid[:, 4:6]] < inst.sec_size[:, None]
    bad = np.flatnonzero(clash.any(1) | unfit | (load[kid[:, 0]] > MAX_LOAD))

    am = np.bincount(inst.sec_course, (kid[:, 2:4] < MORNING).sum(1), minlength=N_COURSES)
    major_am = am >= SEC_PER_COURSE  # course has at least half of its 16 sessions in the morning
    soft_mode = len(bad) == 0
    if soft_mode:
        bad = np.flatnonzero(((kid[:, 2:4] < MORNING) != major_am[inst.sec_course, None]).any(1))
        if len(bad) == 0:
            return
    # Rooms ordered by size so the first free fitting room wastes the least capacity.
    by_size = np.argsort(inst.room_size, kind="stable")

    for i in rng.choice(bad, min(cfg.repairs, len(bad)), replace=False):
        p, need = kid[i, 0], inst.sec_size[i]
        room_occ[kid[i, 4:6], t[i]] -= 1
        prof_occ[p, t[i]] -= 1
        if load[p] > MAX_LOAD or unfit[i]:  # hand it to the least-loaded qualified professor
            q = inst.qualified[inst.sec_course[i]]
            load[p] -= 2
            p = q[np.argmin(load[q] + rng.random(len(q)))]
            load[p] += 2
            kid[i, 0] = p
        fits = by_size[inst.room_size[by_size] >= need]
        best = None
        for k in range(cfg.time_tries):
            if k == 0 and not soft_mode:  # first try: keep the current time, only fix rooms
                dp, sa, sb = kid[i, 1:4]
            else:
                half = 0 if major_am[inst.sec_course[i]] else MORNING
                dp = rng.integers(len(DAY_PAIRS))
                sa, sb = half + rng.integers(MORNING, size=2)
            tt = DAY_PAIRS[dp] * N_SLOTS + (sa, sb)
            if prof_occ[p, tt].any():
                continue
            free = [fits[room_occ[fits, x] == 0] for x in tt]
            if len(free[0]) and len(free[1]):
                best = dp, sa, sb, free[0][0], free[1][0]
                break
        if best is not None:
            kid[i, 1:6] = best
            t[i] = DAY_PAIRS[best[0]] * N_SLOTS + best[1:3]
        room_occ[kid[i, 4:6], t[i]] += 1
        prof_occ[p, t[i]] += 1


def run(inst, cfg=GAConfig(), log=print):
    rng = np.random.default_rng(cfg.seed)
    pop = random_population(inst, rng, cfg.pop_size)
    evals = [evaluate(ind, inst) for ind in pop]
    pen = np.array([e[0] for e in evals])
    hist = History()
    best_pen, stale, t0 = np.inf, 0, time.time()
    n_child = cfg.pop_size - cfg.elite

    # One extra pass records the final population; the loop breaks before breeding from it.
    for gen in range(cfg.generations + 1):
        b = int(pen.argmin())
        terms = evals[b][1]
        hist.best.append(int(pen[b]))
        hist.mean.append(float(pen.mean()))
        hist.hard.append(terms["H"])
        hist.soft.append(terms["S"])
        if pen[b] < best_pen:
            best_pen, stale = pen[b], 0
        else:
            stale += 1
        if gen % 100 == 0:
            log(f"gen {gen:5d}  best={pen[b]:8d}  H={terms['H']:4d}  S={terms['S']:5d}  "
                f"mean={pen.mean():10.1f}  {time.time() - t0:6.1f}s")
        if gen == cfg.generations or (terms["H"] == 0 and stale >= cfg.patience):
            break

        # Tournament selection, vectorised: for each child pick two parents.
        cand = rng.integers(0, cfg.pop_size, (n_child, 2, cfg.tournament))
        parents = np.take_along_axis(cand, pen[cand].argmin(2)[..., None], 2)[..., 0]
        p1, p2 = parents[:, 0], parents[:, 1]

        # Uniform crossover on whole section rows, so a section's genes stay consistent.
        take1 = rng.random((n_child, N_SECTIONS)) < 0.5
        take1[rng.random(n_child) >= cfg.crossover_rate] = True
        kids = np.where(take1[..., None], pop[p1], pop[p2])

        # Random mutation draws replacement genes from a fresh constraint-aware random population,
        # so every mutated value is individually valid (qualified professor, big-enough room).
        fresh = random_population(inst, rng, n_child)
        kids = np.where(rng.random(kids.shape) < cfg.mutation_rate, fresh, kids)
        for kid in kids:
            repair(kid, inst, rng, cfg)

        elite = np.argsort(pen)[:cfg.elite]
        new_evals = [evaluate(k, inst) for k in kids]
        evals = [evals[i] for i in elite] + new_evals
        pop = np.concatenate([pop[elite], kids])
        pen = np.array([e[0] for e in evals])

    b = int(pen.argmin())
    log(f"done: gen {gen}  best={pen[b]}  {evals[b][1]}  {time.time() - t0:.1f}s")
    return pop[b], evals[b], hist
