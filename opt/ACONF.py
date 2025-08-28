from testsetclass import TestsetClass


class ACONF(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['H_g+t+g-', 'P_GG', 'H_tgt', 'P_TT', 'H_ttt']
        self.exp_values = {'2': 0.961, '5': 0.604, '8': 1.302}
        self.testset_calculations = {'2': "-1 * ['P_TT'] +1 * ['P_GG']",
              '5': "-1 * ['H_ttt'] +1 * ['H_tgt']",
              '8': "-1 * ['H_ttt'] +1 * ['H_g+t+g-']"}
        self.charges = {'H_g+t+g-': 0.0, 'P_GG': 0.0, 'H_tgt': 0.0, 'P_TT': 0.0, 'H_ttt': 0.0}
    