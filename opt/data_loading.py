"""
All I/O: reading grid points, assembling reaction objects,
and injecting D4 dispersion energies.
"""
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf

from auxiliary_functions import ( read_molecule, extract_mols,
                                  unstack_array, build_selected_feature_matrices)
from dispersion import inject_dispersion_energies

Reaction = Tuple[str, str, str, tf.Tensor, Dict]  # type alias


def make_feature_matrices(raw28: np.ndarray, features) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw 28×N feature matrix to the two model inputs:
    (2N, n_feat)  and  (2N, 13).
    """

    (ra, gax, gay, gaz, la, ta, hxxa, hxya, hxza, hyya, hyza, hzza,
     rb, gbx, gby, gbz, lb, tb, hxxb, hxyb, hxzb, hyyb, hyzb, hzzb,
     ea, eb, easr, ebsr) = unstack_array(raw28, axis=1)

    gaa = gax * gax + gay * gay + gaz * gaz
    gab = gax * gbx + gay * gby + gaz * gbz
    gbb = gbx * gbx + gby * gby + gbz * gbz

    blocks = [
        (ra, rb),
        (gaa, gab, gbb),
        (ea, eb),
        (ta, tb),
        (la, lb),
        (easr, ebsr)
    ]

    ab_ba = build_selected_feature_matrices(blocks, features)
    x_main = ab_ba.T                              # (2N, n_feat)
    x_feat = build_selected_feature_matrices(blocks, [True]*6).T  # (2N, 13)
    return x_main, x_feat


def build_reaction_list(testset_objects: Dict, a1, a2, TESTSET_NAMES) -> List[Reaction]:
    """
    Iterate over every test-set and build the flat REACTIONS list expected
    by the training loop.
    """
    reactions: List[Reaction] = []
    for ts_name in TESTSET_NAMES:
        ts = testset_objects[ts_name]

        # Compute and attach D4 dispersion contributions (in-place)
        inject_dispersion_energies(ts, ts_name, a1, a2,
                                   atoms=(ts_name == 'FracP'))

        for key, expr in ts.testset_calculations.items():
            reactions.append(
                (
                    ts_name,
                    key,
                    expr,
                    tf.constant(ts.exp_values[key], dtype=tf.float32),
                    ts.dispersion_energy_results
                )
            )

    return reactions


def read_grid_points_for_molecule(ts_name: str, molecule: str, TESTSET_PATHS, MP2):
    """
    Wrapper around `read_molecule` to unify dtype handling
    and slice the first 28 columns expected by `make_feature_matrices`.
    """
    e1e, feat, w, mp2opp, mp2par = read_molecule(
        path=TESTSET_PATHS[ts_name],
        molecule=molecule,
        MP2=MP2
    )
    feat = feat[:28, :]
    return e1e, feat, w, mp2opp, mp2par
