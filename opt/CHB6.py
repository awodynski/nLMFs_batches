from testsetclass import TestsetClass


class CHB6(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['22', '22A', '22B']
        self.exp_values = {'0': -34.43}
        self.testset_calculations = {'0': "+1 * ['22'] -1 * ['22A'] -1 * ['22B']"}
        self.charges = {'22': 1.0, '22A': 1.0, '22B': 0.0}
    