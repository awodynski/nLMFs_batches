import numpy as np
import csv
from debugtest import debugtest
from BH76 import BH76
from W417 import W417
from functools import lru_cache


"""
This module provides auxiliary functions for handling testset data,
assembling features for machine learning, and exporting neural network
weights in a TurboMole-friendly format.

Functions
---------
- _unstack_array(a, axis):
    Moves the specified axis of array 'a' to axis 0.

- read_molecule(path, molecule, MP2):
    Reads grid data from .npy files (features, weights, e1e+coulomb, optionally 
    MP2 info) for a given molecule.

- build_feature_matrices(blocks, logical_mask):
    Helper to build alpha-beta & beta-alpha stacked arrays based on
    `logical_mask`.

- save_weights_turbomole(model, output_path='weights2.csv'):
    Extracts neural network weights/biases and writes them to a CSV file in a
    TurboMole-friendly single-column format.

- extract_mols(expr):
    Return ordered list of unique molecules appearing in `expr`.
"""

def unstack_array(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Moves the specified axis of array 'a' to axis 0.

    Parameters
    ----------
    a : np.ndarray
        The input array.
    axis : int
        The axis to move to the front.

    Returns
    -------
    np.ndarray
        A new array with the selected axis moved to 0.
    """
    return np.moveaxis(a, axis, 0)

def read_molecule(path: str, molecule: str, MP2: bool
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray | None, np.ndarray | None, ]:
    """
    Reads grid data for the specified molecule from .grid files.

    The function loads:
      - features: (N,  ...), shape read from `<molecule>_features.grid`
      - weights:  (N,  ...), shape read from `<molecule>_weights.grid`
      - e_1e:     scalar energy from `<molecule>_e1e.grid` includes 1e energy +
                  2-electron Coulomb interaction
      - optional MP2 data: e_mp2opp (total energy not density)
                           e_mp2par (total energy not density),
                           if MP2=True.

    Parameters
    ----------
    path : str
        Base path to the directory containing .grid files.
    molecule : str
        The molecule name (file prefix).
    MP2 : bool
        If True, also load e_mp2opp and e_mp2par.

    Returns
    -------
    tuple
        (e_1e, features, weights, e_mp2opp, e_mp2par)
    """
    # Build the base file name
    base_name = path + molecule

    # Load features
    fname = base_name + "_features.npy"
#    features = (np.loadtxt(fname, dtype=float).T).astype(np.float32)
    features = np.load(fname, mmap_mode='r')
    # Load weights
    fname = base_name + "_weights.npy"
#    weights = (np.loadtxt(fname, dtype=float).T).astype(np.float32)
    weights =  np.load(fname, mmap_mode='r')

    # Load e_1e
    fname = base_name + "_e1e.npy"
    e_1e =  np.load(fname, mmap_mode='r')
#    e_1e =  (np.loadtxt(fname, dtype=float).T).astype(np.float32)

    # If MP2 data is requested, load e_mp2opp and e_mp2par
    if MP2:
        fname = base_name + "_emp2opp.npy"
 #       e_mp2opp = (np.loadtxt(fname, dtype=float).T).astype(np.float32)
        e_mp2opp = np.load(fname, mmap_mode='r')

        fname = base_name + "_emp2par.npy"
#        e_mp2par = (np.loadtxt(fname, dtype=float).T).astype(np.float32)
        e_mp2par = np.load(fname, mmap_mode='r')
    else:
        e_mp2opp, e_mp2par = None, None

    return e_1e, features, weights, e_mp2opp, e_mp2par



def build_selected_feature_matrices(blocks, logical_mask):
    """Helper to build alpha-beta & beta-alpha stacked arrays based on
    `logical_mask`."""
    ab_list = []
    ba_list = []
    for (idx, block) in enumerate(blocks):
        # If we want this block according to logical_features
        if logical_mask[idx]:
            # alpha-beta as is
            ab_list.extend(block)
            # beta-alpha is reversed
            ba_list.extend(block[::-1])

    # Combine
    ab_array = np.array(ab_list)
    ba_array = np.array(ba_list)
    return np.concatenate((ab_array, ba_array), axis=1)



def save_weights_turbomole(model, output_path: str = 'weights2.csv') -> None:
    """
    Extracts model weights and biases, then writes them to a CSV file
    in a TurboMole-friendly, single-column format.

    Parameters
    ----------
    model : tf.keras.Model
        The trained neural network model containing layers with weights & biases.
    output_path : str, optional
        The filename to write the CSV data to.
    """
    weights_and_biases = [layer.get_weights() for layer in model.layers]

    # Flatten all weights in single column format
    single_column_values = []

    for layer_index, layer_weights in enumerate(weights_and_biases, start=1):
        if len(layer_weights) == 2:
            w, b = layer_weights
            # Transpose weights so columns match nodes
            w_T = w.T
            # Flatten row by row (or col by col, depends on how you want it)
            for row in w_T:
                single_column_values.extend(row)
            # Then biases
            single_column_values.extend(b)
        else:
            print(f"Layer {layer_index} does not contain weights or biases.")

    # Write everything into one CSV column
    with open(output_path, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for val in single_column_values:
            writer.writerow([f"{val:24.17E}"])


"""
Tiny regex helpers shared across modules.
"""
import re
from typing import List

_EXTRACT_RE = re.compile(r"\['([^']+)'\]")

def extract_mols(expr: str) -> List[str]:
    """Return ordered list of unique molecules appearing in `expr`."""
    seen, order = set(), []
    for m in _EXTRACT_RE.findall(expr):
        if m not in seen:
            seen.add(m)
            order.append(m)
    return order
