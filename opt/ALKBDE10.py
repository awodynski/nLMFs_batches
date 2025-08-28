from testsetclass import TestsetClass


class ALKBDE10(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['be', 'bef', 'f']
        self.exp_values = {'0': 138.7}
        self.testset_calculations = {'0': "-1 * ['bef'] +1 * ['be'] +1 * ['f']"}
        self.charges = {'be': 0.0, 'bef': 0.0, 'f': -0.0}
    