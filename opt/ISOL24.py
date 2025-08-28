from testsetclass import TestsetClass


class ISOL24(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['i13e', 'i13p']
        self.exp_values = {'12': 33.02}
        self.testset_calculations = {'12': "-1 * ['i13e'] +1 * ['i13p']"}
        self.charges = {'i13e': 0.0, 'i13p': 0.0}
    