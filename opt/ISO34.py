from testsetclass import TestsetClass


class ISO34(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['E7', 'P7', 'E12', 'P12']
        self.exp_values = {'6': 11.15, '11': 45.65}
        self.testset_calculations = {'6': "-1 * ['E7'] +1 * ['P7']",
              '11': "-1 * ['E12'] +1 * ['P12']"}
        self.charges = {'E7': 0.0, 'P7': 0.0, 'E12': 0.0, 'P12': 0.0}
    