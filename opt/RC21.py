from testsetclass import TestsetClass


class RC21(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['2e', '2p2', '2p3']
        self.exp_values = {'3': 59.44}
        self.testset_calculations = {'3': "-1 * ['2e'] +1 * ['2p2'] +1 * ['2p3']"}
        self.charges = {'2e': 1.0, '2p2': 0.0, '2p3': 1.0}
    