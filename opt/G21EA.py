from testsetclass import TestsetClass


class G21EA(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['EA_p', 'EA_p-', 'EA_21n', 'EA_21', 'EA_25n', 'EA_25']
        self.exp_values = {'4': 17.3,'20': -0.2,'24': 54.7}
        self.testset_calculations = {'4': "+1 * ['EA_p'] -1 * ['EA_p-']",
              '20': "+1 * ['EA_21n'] -1 * ['EA_21']",
              '24': "+1 * ['EA_25n'] -1 * ['EA_25']"}
        self.charges = {'EA_p': -0.0, 'EA_p-': -1.0, 'EA_21n': 0.0, 'EA_21': -1.0, 'EA_25n': -0.0, 'EA_25': -1.0}
    