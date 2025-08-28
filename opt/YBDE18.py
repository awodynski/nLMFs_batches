from testsetclass import TestsetClass


class YBDE18(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['f2s-cbh22', 'f2s', 'cbh22']
        self.exp_values = {'0': 57.17}
        self.testset_calculations = {'0': "-1 * ['f2s-cbh22'] +1 * ['f2s'] +1 * ['cbh22']"}
        self.charges = {'f2s-cbh22': -0.0, 'f2s': -0.0, 'cbh22': -0.0}
    