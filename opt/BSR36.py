from testsetclass import TestsetClass


class BSR36(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['c2h6', 'h4', 'ch4', 'r6']
        self.exp_values = {'3': 8.85, '20': 9.78}
        self.testset_calculations = {'3': "+5 * ['c2h6'] -1 * ['h4'] -4 * ['ch4']",
              '20': "+7 * ['c2h6'] -1 * ['r6'] -7 * ['ch4']"}
        self.charges = {'c2h6': 0.0, 'h4': 0.0, 'ch4': 0.0, 'r6': 0.0}
    