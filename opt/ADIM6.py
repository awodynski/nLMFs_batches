from testsetclass import TestsetClass


class ADIM6(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['AM3', 'AD3']
        self.exp_values = {'1': 1.99}
        self.testset_calculations = {'1': "+2 * ['AM3'] -1 * ['AD3']"}
        self.charges = {'AM3': 0.0, 'AD3': 0.0}
    