"""
Injects D4 dispersion energies into any test-set object that implements
the appropriate interface.
"""
def inject_dispersion_energies(testset, name: str,
                               a1: float, a2: float,
                               atoms: bool = False) -> None:
    """
    Computes and writes `testset.dispersion_energy_results` in-place.

    Parameters
    ----------
    atoms : bool
        Set to True for FracP (atomic reference energies).
    """
    testset.calculate_dispersion_energies(name, a1, a2, atoms)
