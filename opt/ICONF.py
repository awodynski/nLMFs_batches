from testsetclass import TestsetClass


class ICONF(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['H4P2O7_1', 'H4P2O7_3']
        self.exp_values = {'15': 3.66}
        self.testset_calculations = {'15': "-1 * ['H4P2O7_1'] +1 * ['H4P2O7_3']"}
        self.charges = {'H4P2O7_1': -0.0, 'H4P2O7_3': -0.0}
    