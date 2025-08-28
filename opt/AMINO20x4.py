from testsetclass import TestsetClass


class AMINO20x4(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['GLN_xai', 'GLN_xap', 'LYS_xap', 'LYS_xat', 'PHE_xaw', 'PHE_xal']
        self.exp_values = {'22': 4.045, '47': 0.540, '55': 1.889}
        self.testset_calculations = {'22': "-1 * ['GLN_xai'] +1 * ['GLN_xap']",
              '47': "-1 * ['LYS_xap'] +1 * ['LYS_xat']",
              '55': "-1 * ['PHE_xaw'] +1 * ['PHE_xal']"}
        self.charges = {'GLN_xai': -0.0, 'GLN_xap': 0.0, 'LYS_xap': -0.0, 'LYS_xat': -0.0, 'PHE_xaw': 0.0, 'PHE_xal': 0.0}
    