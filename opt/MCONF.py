from testsetclass import TestsetClass


class MCONF(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['1', '8', '23', '31', '43']
        self.exp_values = {'6': 2.92,'21': 5.31, '29': 4.86, '41': 7.32}
        self.testset_calculations = {'6': "-1 * ['1'] +1 * ['8']",
              '21': "-1 * ['1'] +1 * ['23']",
              '29': "-1 * ['1'] +1 * ['31']",
              '41': "-1 * ['1'] +1 * ['43']"}
        self.charges = {'1': 0.0, '8': 0.0, '23': 0.0, '31': 0.0, '43': 0.0}
    