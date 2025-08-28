from testsetclass import TestsetClass


class RSE43(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['E30', 'E32', 'E45', 'P1', 'E1', 'P30', 'P32', 'P45']
        self.exp_values = {'27': -6.0,'29': -4.3,'42': -2.3}
        self.testset_calculations = {'27': "-1 * ['E30'] -1 * ['P1'] +1 * ['E1']+1 * ['P30']",
            '29': "-1 * ['E32'] -1 * ['P1'] +1 * ['E1']+1 * ['P32']",
            '42': "-1 * ['E45'] -1 * ['P1'] +1 * ['E1']+1 * ['P45']"}
        self.charges = {'E30': 0.0, 'E32': 0.0, 'E45': 0.0, 'P1': 0.0, 'E1': 0.0, 'P30': -0.0, 'P32': 0.0, 'P45': 0.0}
    