"""Fitness function. A chromosome is an int array (N_SECTIONS, 6):
[prof, day_pair, slot_a, slot_b, room_a, room_b] per section."""
import numpy as np

from instance import (DAY_PAIRS, MAX_LOAD, MORNING, N_COURSES, N_DAYS, N_PROFS, N_ROOMS,
                      N_SLOTS, N_TIMES, SEC_PER_COURSE)

PROF, DAYPAIR, SLOT_A, SLOT_B, ROOM_A, ROOM_B = range(6)

W_HARD = 1000  # any hard violation outweighs every possible soft improvement
SOFT_WEIGHTS = {
    "prof_gap": 1,  # S1: idle timeslots between a professor's first and last session of a day
    "prof_mixed_day": 2,  # S2: professor-days that contain both morning and afternoon sessions
    "course_split": 1,  # S3: sessions of a course in its minority half of the day
    "room_waste": 1,  # S4: sum of (room size - registration size)
    "no_favorite": 5,  # Bonus 1: professors who teach none of their favourite courses
}
HARD_TERMS = ["room_clash", "prof_clash", "prof_overload", "room_too_small", "prof_unqualified"]


def sessions(ind):
    """Flatten sections into sessions; session 2*i and 2*i+1 belong to section i."""
    prof = np.repeat(ind[:, PROF], 2)
    day = DAY_PAIRS[ind[:, DAYPAIR]].ravel()
    slot = ind[:, SLOT_A:SLOT_B + 1].ravel()
    room = ind[:, ROOM_A:ROOM_B + 1].ravel()
    return prof, day, slot, room


def evaluate(ind, inst):
    """Return (penalty, terms dict)."""
    prof, day, slot, room = sessions(ind)
    t = day * N_SLOTS + slot
    size = np.repeat(inst.sec_size, 2)
    course = np.repeat(inst.sec_course, 2)
    sec_prof = ind[:, PROF]

    room_key = room * N_TIMES + t
    room_cnt = np.bincount(room_key, minlength=N_ROOMS * N_TIMES)
    prof_key = prof * N_TIMES + t
    prof_cnt = np.bincount(prof_key, minlength=N_PROFS * N_TIMES)
    load = 2 * np.bincount(sec_prof, minlength=N_PROFS)
    rsize = inst.room_size[room]
    unqualified = ~inst.can_teach[sec_prof, inst.sec_course]

    terms = {
        "room_clash": int(np.maximum(room_cnt - 1, 0).sum()),
        "prof_clash": int(np.maximum(prof_cnt - 1, 0).sum()),
        "prof_overload": int(np.maximum(load - MAX_LOAD, 0).sum()),
        "room_too_small": int((size > rsize).sum()),
        "prof_unqualified": int(unqualified.sum()),
    }

    occ = (prof_cnt > 0).reshape(N_PROFS, N_DAYS, N_SLOTS)
    n = occ.sum(2)
    first = occ.argmax(2)
    last = N_SLOTS - 1 - occ[:, :, ::-1].argmax(2)
    terms["prof_gap"] = int(np.where(n > 0, last - first + 1 - n, 0).sum())
    terms["prof_mixed_day"] = int((occ[:, :, :MORNING].any(2) & occ[:, :, MORNING:].any(2)).sum())
    am = np.bincount(course, weights=slot < MORNING, minlength=N_COURSES)
    terms["course_split"] = int(np.minimum(am, 2 * SEC_PER_COURSE - am).sum())
    terms["room_waste"] = int(np.maximum(rsize - size, 0).sum())
    taught = np.zeros((N_PROFS, N_COURSES), bool)
    taught[sec_prof, inst.sec_course] = True
    terms["no_favorite"] = int((~(taught & inst.favorite).any(1)).sum())

    hard = sum(terms[k] for k in HARD_TERMS)
    soft = sum(w * terms[k] for k, w in SOFT_WEIGHTS.items())
    terms["H"], terms["S"] = hard, soft
    return W_HARD * hard + soft, terms


def fitness(penalty):
    return 1.0 / (1.0 + penalty)
