from testsetclass import TestsetClass


class RG18(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['ne2', 'ne', 'c2h6Ne', 'ne', 'c2h6']
        self.exp_values = {'0': 0.08, '14': 0.24}
        self.testset_calculations = {'0': "-1 * ['ne2'] +2 * ['ne']",
              '14': "-1 * ['c2h6Ne'] +1 * ['ne'] +1 * ['c2h6']"}
        self.charges = {'ne2': -0.0, 'ne': 0.0, 'c2h6Ne': -0.0, 'c2h6': 0.0}
    