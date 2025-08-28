from testsetclass import TestsetClass


class UPU23(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['2p', '1p']
        self.exp_values = {'4': 2.02}
        self.testset_calculations = {'4': "-1 * ['2p'] +1 * ['1p']"}
        self.charges = {'2p': -1.0, '1p': -1.0}
    