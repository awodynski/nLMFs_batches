from testsetclass import TestsetClass


class BH76RC(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['ch3', 'H2', 'CH4', 'h']
        self.exp_values = {'15': -3.11}
        self.testset_calculations = {'15': "-1 * ['ch3'] -1 * ['H2'] +1 * ['CH4'] +1 * ['h']"}
        self.charges = {'ch3': 0.0, 'H2': 0.0, 'CH4': -0.0, 'h': -0.0}
    