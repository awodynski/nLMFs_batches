from testsetclass import TestsetClass


class HAL59(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['PCH_PhBr', 'PCH', 'PhBr', 'NCH_PhI', 'NCH', 'PhI', 'NH3_FCl', 'NH3', 'FCl', 'BrBr_OCH2', 'BrBr', 'OCH2', 'MeI_pyr', 'MeI', 'pyr', '27_CH3Br-benA', '27_CH3Br-benB', '27_CH3Br-benAB']
        self.exp_values = {'0': 0.85, '7': 1.87, '20': 10.54, '37': 4.41, '42': 3.61, '55': 1.81}
        self.testset_calculations = {'0': "-1 * ['PCH_PhBr'] +1 * ['PCH'] +1 * ['PhBr']",
              '7': "-1 * ['NCH_PhI'] +1 * ['NCH'] +1 * ['PhI']",
              '20': "-1 * ['NH3_FCl'] +1 * ['NH3'] +1 * ['FCl']",
              '37': "-1 * ['BrBr_OCH2'] +1 * ['BrBr'] +1 * ['OCH2']",
              '42': "-1 * ['MeI_pyr'] +1 * ['MeI'] +1 * ['pyr']",
              '55': "+1 * ['27_CH3Br-benA'] +1 * ['27_CH3Br-benB'] -1 * ['27_CH3Br-benAB']"}
        self.charges = {'PCH_PhBr': -0.0, 'PCH': -0.0, 'PhBr': -0.0, 'NCH_PhI': -0.0, 'NCH': -0.0, 'PhI': 0.0, 'NH3_FCl': -0.0, 'NH3': -0.0, 'FCl': -0.0, 'BrBr_OCH2': 0.0, 'BrBr': 0.0, 'OCH2': 0.0, 'MeI_pyr': 0.0, 'MeI': 0.0, 'pyr': 0.0, '27_CH3Br-benA': 0.0, '27_CH3Br-benB': 0.0, '27_CH3Br-benAB': 0.0}
    