from testsetclass import TestsetClass


class SCONF(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['C1', 'C11']
        self.exp_values = {'9': 5.65}
        self.testset_calculations = {'9': "-1 * ['C1'] +1 * ['C11']"}
        self.charges = {'C1': -0.0, 'C11': -0.0}
    