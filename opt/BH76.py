from testsetclass import TestsetClass


class BH76(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['hcl', 'h', 'hclhts', 'ch3cl', 'cl-', 'clch3clts', 'H2O', 'C2H5', 'RKT09', 'NH3', 'ch3', 'RKT21']
        self.exp_values = {'5': 17.8, '17': 2.5, '53': 20.4, '73': 16.9}
        self.testset_calculations = {'5': "-1 * ['hcl'] -1 * ['h'] +1 * ['hclhts']",
              '17': "-1 * ['ch3cl'] -1 * ['cl-'] +1 * ['clch3clts']",
              '53': "-1 * ['H2O'] -1 * ['C2H5'] +1 * ['RKT09']",
              '73': "-1 * ['NH3'] -1 * ['ch3'] +1 * ['RKT21']"}
        self.charges = {'hcl': 0.0, 'h': -0.0, 'hclhts': -0.0, 'ch3cl': -0.0, 'cl-': -1.0, 'clch3clts': -1.0, 'H2O': 0.0, 'C2H5': 0.0, 'RKT09': 0.0, 'NH3': -0.0, 'ch3': 0.0, 'RKT21': 0.0}
    