from testsetclass import TestsetClass


class PArel(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['T1', 'T0', 'T3', 'sugar0', 'sugar2', 'c2cl43', 'c2cl41']
        self.exp_values = {'2': 2.94, '4': 7.19, '9': 2.98, '19': 2.15}
        self.testset_calculations = {'2': "-1 * ['T1'] +1 * ['T0']",
              '4': "-1 * ['T1'] +1 * ['T3']",
              '9': "-1 * ['sugar0'] +1 * ['sugar2']",
              '19': "-1 * ['c2cl43'] +1 * ['c2cl41']"}
        self.charges = {'T1': 1.0, 'T0': 1.0, 'T3': 1.0, 'sugar0': 1.0, 'sugar2': 1.0, 'c2cl43': 1.0, 'c2cl41': 1.0}
    