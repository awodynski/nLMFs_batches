from testsetclass import TestsetClass


class PCONF21(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['99', '412', '691', 'SER_ab', 'SER_b']
        self.exp_values = {'6': 2.18, '7': 1.61, '17': 2.74}
        self.testset_calculations = {'6': "-1 * ['99'] +1 * ['412']",
              '7': "-1 * ['99'] +1 * ['691']",
              '17': "-1 * ['SER_ab'] +1 * ['SER_b']"}
        self.charges = {'99': -0.0, '412': -0.0, '691': -0.0, 'SER_ab': 0.0, 'SER_b': 0.0}
    