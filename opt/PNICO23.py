from testsetclass import TestsetClass


class PNICO23(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['13', '13a', '13b', '17', '17a', '17b']
        self.exp_values = {'12': 3.98,'16': 7.10}
        self.testset_calculations = {'12': "-1 * ['13'] +1 * ['13a'] +1 * ['13b']",
              '16': "-1 * ['17'] +1 * ['17a'] +1 * ['17b']"}
        self.charges = {'13': -0.0, '13a': -0.0, '13b': -0.0, '17': 0.0, '17a': 0.0, '17b': 0.0}
    