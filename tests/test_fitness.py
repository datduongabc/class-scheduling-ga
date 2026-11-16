"""The vectorised fitness is checked against a deliberately naive, loop-based re-implementation
of every constraint, plus hand-crafted cases for each hard constraint."""
from collections import Counter, defaultdict

import numpy as np
import pytest

from fitness import evaluate
from ga import GAConfig, random_population, repair, run
from instance import DAY_PAIRS, ROOM_MIX, is_feasible_capacity, make_instance

INST = make_instance(42)


def naive_terms(ind, inst):
    room_t, prof_t, load = Counter(), Counter(), Counter()
    prof_day = defaultdict(list)
    course_am = Counter()
    taught = set()
    t = {k: 0 for k in ["room_too_small", "prof_unqualified", "room_waste"]}
    for s, (p, dp, sa, sb, ra, rb) in enumerate(ind):
        c = inst.sec_course[s]
        taught.add((p, c))
        load[p] += 2
        if not inst.can_teach[p, c]:
            t["prof_unqualified"] += 1
        d1, d2 = DAY_PAIRS[dp]
        assert abs(d1 - d2) >= 2  # hard constraint 2 holds by construction
        for d, sl, r in ((d1, sa, ra), (d2, sb, rb)):
            room_t[r, d, sl] += 1
            prof_t[p, d, sl] += 1
            prof_day[p, d].append(sl)
            course_am[c] += sl < 3
            if inst.sec_size[s] > inst.room_size[r]:
                t["room_too_small"] += 1
            t["room_waste"] += max(inst.room_size[r] - inst.sec_size[s], 0)
    t["room_clash"] = sum(v - 1 for v in room_t.values() if v > 1)
    t["prof_clash"] = sum(v - 1 for v in prof_t.values() if v > 1)
    t["prof_overload"] = sum(max(v - 20, 0) for v in load.values())
    t["prof_gap"] = sum(max(s) - min(s) + 1 - len(set(s)) for s in prof_day.values())
    t["prof_mixed_day"] = sum(min(s) < 3 <= max(s) for s in prof_day.values())
    t["course_split"] = sum(min(course_am[c], 16 - course_am[c]) for c in range(60))
    t["no_favorite"] = sum(not any(inst.favorite[p, c] for (q, c) in taught if q == p)
                           for p in range(60))
    return t


def base():
    """A clash-free schedule built from repeated repair, used to inject single violations."""
    best, _, _ = run(INST, GAConfig(pop_size=20, generations=400, patience=1, seed=1), log=lambda *_: 0)
    assert evaluate(best, INST)[1]["H"] == 0
    return best


@pytest.fixture(scope="module")
def feasible():
    return base()


@pytest.mark.parametrize("seed", range(5))
def test_vectorised_matches_naive(seed):
    rng = np.random.default_rng(seed)
    ind = random_population(INST, rng, 1)[0]
    ind[:, 0] = rng.integers(0, 60, len(ind))  # also exercise unqualified professors
    _, terms = evaluate(ind, INST)
    for k, v in naive_terms(ind, INST).items():
        assert terms[k] == v, k


def test_naive_matches_on_feasible(feasible):
    _, terms = evaluate(feasible, INST)
    for k, v in naive_terms(feasible, INST).items():
        assert terms[k] == v, k


def test_room_clash(feasible):
    ind = feasible.copy()
    ind[1, 1:4], ind[1, 4:6] = ind[0, 1:4], ind[0, 4:6]  # section 1 in section 0's rooms/times
    ind[1, 0] = ind[0, 0]
    _, terms = evaluate(ind, INST)
    assert terms["room_clash"] >= 2
    repair(ind, INST, np.random.default_rng(0), GAConfig())  # repair finds the clash by itself
    assert evaluate(ind, INST)[1]["H"] < terms["H"]


def test_prof_clash_and_overload(feasible):
    ind = feasible.copy()
    ind[:, 0] = INST.qualified[0][0]  # one professor teaches everything (and is unqualified)
    _, terms = evaluate(ind, INST)
    assert terms["prof_clash"] > 0
    assert terms["prof_overload"] == 2 * len(ind) - 20
    assert terms["prof_unqualified"] == (~INST.can_teach[ind[0, 0], INST.sec_course]).sum()


def test_room_too_small(feasible):
    ind = feasible.copy()
    big = np.flatnonzero(INST.sec_size == 2)[0]
    small = np.flatnonzero(INST.room_size == 0)[0]
    ind[big, 4] = small
    _, terms = evaluate(ind, INST)
    assert terms["room_too_small"] >= 1


def test_repair_never_adds_hard_violations(feasible):
    rng = np.random.default_rng(0)
    ind = feasible.copy()
    _, before = evaluate(ind, INST)
    repair(ind, INST, rng, GAConfig(repairs=50))  # soft mode: no hard conflicts
    _, after = evaluate(ind, INST)
    assert after["H"] == 0 and after["course_split"] <= before["course_split"]


def test_instance_properties():
    assert (INST.can_teach.sum(1) == 5).all()  # every professor has 5 distinct courses
    assert (INST.can_teach.sum(0) == 5).all()  # every course has 5 qualified professors
    assert (INST.favorite.sum(1) == 3).all() and not (INST.favorite & ~INST.can_teach).any()
    assert (np.bincount(INST.room_size, minlength=3) == ROOM_MIX).all()
    assert is_feasible_capacity(INST.room_size, INST.sec_size)


def test_capacity_check_rejects_too_few_large_rooms():
    rooms = np.zeros(35, int)
    rooms[0] = 2  # one Large room: 30 Large room-timeslots
    secs = np.full(480, 2)  # 960 Large sessions
    assert not is_feasible_capacity(rooms, secs)


def test_zero_generations_does_not_crash():
    _, (_, terms), hist = run(INST, GAConfig(pop_size=4, generations=0), log=lambda *_: 0)
    assert len(hist.best) == 1 and terms["H"] == hist.hard[0]
