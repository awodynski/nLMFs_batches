from testsetclass import TestsetClass


class BHROT27(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['ethane_st', 'ethane_ecl', 'h2s2', 'h2s2_cis', 'bithiophene_anti', 'bithiophene_TS']
        self.exp_values = {'0': 2.73, '9': 8.03, '19': 1.78}
        self.testset_calculations = {'0': "-1 * ['ethane_st'] +1 * ['ethane_ecl']",
              '9': "-1 * ['h2s2'] +1 * ['h2s2_cis']",
              '19': "-1 * ['bithiophene_anti'] +1 * ['bithiophene_TS']"}
        self.charges = {'ethane_st': 0.0, 'ethane_ecl': 0.0, 'h2s2': 0.0, 'h2s2_cis': -0.0, 'bithiophene_anti': 0.0, 'bithiophene_TS': 0.0}
    