from testsetclass import TestsetClass


class HEAVY28(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['bih3_hbr', 'bih3', 'hbr', 'teh2_h2s', 'teh2', 'h2s']
        self.exp_values = {'4': 0.98, '23': 0.48}
        self.testset_calculations = {'4': "-1 * ['bih3_hbr'] +1 * ['bih3'] +1 * ['hbr']",
              '23': "-1 * ['teh2_h2s'] +1 * ['teh2'] +1 * ['h2s']"}
        self.charges = {'bih3_hbr': 0.0, 'bih3': 0.0, 'hbr': -0.0, 'teh2_h2s': 0.0, 'teh2': 0.0, 'h2s': -0.0}
    