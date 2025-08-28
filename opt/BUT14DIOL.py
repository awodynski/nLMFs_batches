from testsetclass import TestsetClass


class BUT14DIOL(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['B1', 'B29', 'B40', 'B64']
        self.exp_values = {'27': 2.83,'38': 3.10,'62': 4.31}
        self.testset_calculations = {'27': "-1 * ['B1'] +1 * ['B29']",
              '38': "-1 * ['B1'] +1 * ['B40']",
              '62': "-1 * ['B1'] +1 * ['B64']"}
        self.charges = {'B1': -0.0, 'B29': 0.0, 'B40': 0.0, 'B64': 0.0}
    