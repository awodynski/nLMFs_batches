from testsetclass import TestsetClass


class FH51(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['methylpyridine', 'H2', 'dimethylpyrrole']
        self.exp_values = {'24': -20.27}
        self.testset_calculations = {'24': "-1 * ['methylpyridine'] -1 * ['H2'] +1 * ['dimethylpyrrole']"}
        self.charges = {'methylpyridine': 0.0, 'H2': 0.0, 'dimethylpyrrole': -0.0}
    