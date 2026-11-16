"""Problem instance: courses, sections, rooms, professors (seeded, reproducible)."""
from dataclasses import dataclass

import numpy as np

N_COURSES = 60
SEC_PER_COURSE = 8
N_SECTIONS = N_COURSES * SEC_PER_COURSE  # 480
N_PROFS = 60
N_DAYS = 5
N_SLOTS = 6
N_TIMES = N_DAYS * N_SLOTS  # 30 (day, slot) cells per week
N_ROOMS = 35
MAX_LOAD = 20  # sessions per professor per week
LIST_LEN = 5  # courses a professor is able to teach
N_FAV = 3  # favourite courses per professor (subset of the list)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]
# The only day pairs that are neither the same day nor adjacent days.
# Encoding a section's days as one of these makes hard constraint 2 impossible to violate.
DAY_PAIRS = np.array([(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)])
MORNING = 3  # slots 0..2 (timeslots 1-3) are morning, 3..5 (timeslots 4-6) afternoon

ROOM_MIX = (13, 12, 10)  # number of Small / Average / Large rooms
SECTION_P = (0.40, 0.35, 0.25)  # probability of registration size Small / Average / Large


@dataclass
class Instance:
    room_size: np.ndarray  # (N_ROOMS,) in {0,1,2}
    sec_course: np.ndarray  # (N_SECTIONS,) course index of each section
    sec_size: np.ndarray  # (N_SECTIONS,) registration size in {0,1,2}
    can_teach: np.ndarray  # (N_PROFS, N_COURSES) bool
    favorite: np.ndarray  # (N_PROFS, N_COURSES) bool, subset of can_teach
    qualified: list  # qualified[c] -> array of professors able to teach course c
    rooms_for_size: list  # rooms_for_size[s] -> array of rooms with size >= s


def course_name(c):
    return f"c{c + 1:02d}"


def prof_name(p):
    return f"p{p + 1:02d}"


def room_name(r):
    return f"rm{r + 1:02d}"


def section_name(s):
    return f"{course_name(s // SEC_PER_COURSE)}-s{s % SEC_PER_COURSE + 1}"


def is_feasible_capacity(room_size, sec_size):
    """Hall-type check: for every size tier, sessions needing >= tier fit in rooms of size >= tier."""
    for tier in (0, 1, 2):
        demand = 2 * np.sum(sec_size >= tier)
        supply = N_TIMES * np.sum(room_size >= tier)
        if demand > supply:
            return False
    return True


def make_instance(seed=0):
    rng = np.random.default_rng(seed)
    room_size = rng.permutation(np.repeat([0, 1, 2], ROOM_MIX))
    sec_course = np.repeat(np.arange(N_COURSES), SEC_PER_COURSE)
    for _ in range(1000):  # a bad ROOM_MIX / SECTION_P would otherwise loop forever
        sec_size = rng.choice(3, size=N_SECTIONS, p=SECTION_P)
        if is_feasible_capacity(room_size, sec_size):
            break
    else:
        raise ValueError("ROOM_MIX cannot host the section sizes drawn from SECTION_P")

    # Randomised cyclic design: professor p can teach courses pi[(sigma[p] + off_k) % 60].
    # This gives every professor 5 distinct courses and every course exactly 5 qualified professors,
    # so the 16 sessions of each course can always be covered without exceeding MAX_LOAD.
    offsets = np.concatenate([[0], rng.choice(np.arange(1, N_COURSES), LIST_LEN - 1, replace=False)])
    pi, sigma = rng.permutation(N_COURSES), rng.permutation(N_PROFS)
    can_teach = np.zeros((N_PROFS, N_COURSES), bool)
    favorite = np.zeros((N_PROFS, N_COURSES), bool)
    for p in range(N_PROFS):
        courses = pi[(sigma[p] + offsets) % N_COURSES]
        can_teach[p, courses] = True
        favorite[p, rng.choice(courses, N_FAV, replace=False)] = True

    return Instance(
        room_size=room_size,
        sec_course=sec_course,
        sec_size=sec_size,
        can_teach=can_teach,
        favorite=favorite,
        qualified=[np.flatnonzero(can_teach[:, c]) for c in range(N_COURSES)],
        rooms_for_size=[np.flatnonzero(room_size >= s) for s in range(3)],
    )
