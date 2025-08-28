from testsetclass import TestsetClass


class AHB21(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['1', '1A', '1B', '15', '15A', '15B']
        self.exp_values = {'0': -17.79, '14': -8.62}
        self.testset_calculations = {'0': "+1 * ['1'] -1 * ['1A'] -1 * ['1B']",
              '14': "+1 * ['15'] -1 * ['15A'] -1 * ['15B']"}
        self.charges = {'1': -1.0, '1A': -1.0, '1B': 0.0, '15': -1.0, '15A': -1.0, '15B': 0.0}
    