from testsetclass import TestsetClass


class DC13(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['carbooxo2', 'carbooxo1', 'o3', 'c2h4', 'o3_c2h4_add']
        self.exp_values = {'5': -25.7, '11': -58.7}
        self.testset_calculations = {'5': "-1 * ['carbooxo2'] +1 * ['carbooxo1']",
              '11': "-1 * ['o3'] -1 * ['c2h4'] +1 * ['o3_c2h4_add']"}
        self.charges = {'carbooxo2': 0.0, 'carbooxo1': 0.0, 'o3': 0.0, 'c2h4': 0.0, 'o3_c2h4_add': -0.0}
    