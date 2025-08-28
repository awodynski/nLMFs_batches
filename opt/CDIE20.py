from testsetclass import TestsetClass


class CDIE20(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['R21', 'P20', 'R48', 'P48']
        self.exp_values = {'1': 4.4, '13': 4.4}
        self.testset_calculations = {'1': "-1 * ['R21'] +1 * ['P20']",
              '13': "-1 * ['R48'] +1 * ['P48']"}
        self.charges = {'R21': 0.0, 'P20': 0.0, 'R48': 0.0, 'P48': -0.0}
    