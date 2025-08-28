from testsetclass import TestsetClass


class G2RC(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['34', '1', '11']
        self.exp_values = {'14': -39.43}
        self.testset_calculations = {'14': "-1 * ['34'] -3 * ['1'] +2 * ['11']"}
        self.charges = {'34': 0.0, '1': 0.0, '11': 0.0}
    