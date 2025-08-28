from testsetclass import TestsetClass


class IL16(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['147', '147A', '147B']
        self.exp_values = {'2': -116.91}
        self.testset_calculations = {'2': "+1 * ['147'] -1 * ['147A'] -1 * ['147B']"}
        self.charges = {'147': 0.0, '147A': 1.0, '147B': -1.0}
    