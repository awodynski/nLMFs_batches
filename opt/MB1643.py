from testsetclass import TestsetClass


class MB1643(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['04', 'H2', 'BH3', 'CH4', 'O2', 'F2', 'AlH3', 'SiH4', '31', 'P2', 'Cl2']
        self.exp_values = {'3': 356.8481,  '30': 534.9003}
        self.testset_calculations = {'3': "-2 * ['04'] -17 * ['H2'] +6 * ['BH3'] +2 * ['CH4'] +1 * ['O2'] +1 * ['F2'] +2 * ['AlH3'] +4 * ['SiH4']",
            '30': "-2 * ['31'] -8 * ['H2'] +4 * ['BH3'] +2 * ['CH4'] +1 * ['O2'] +1 * ['F2'] +2 * ['AlH3'] +4 * ['P2'] +1 * ['Cl2']"}
        self.charges = {'04': -0.0, 'H2': -0.0, 'BH3': -0.0, 'CH4': -0.0, 'O2': -0.0, 'F2': -0.0, 'AlH3': -0.0, 'SiH4': -0.0, '31': 0.0, 'P2': -0.0, 'Cl2': -0.0}
    