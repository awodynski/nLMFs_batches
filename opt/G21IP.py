from testsetclass import TestsetClass


class G21IP(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['IP_72', '26']
        self.exp_values = {'27': 243.709}
        self.testset_calculations = {'27': "+1 * ['IP_72'] -1 * ['26']"}
        self.charges = {'IP_72': 1.0, '26': -0.0}
    