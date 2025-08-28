from testsetclass import TestsetClass


class CARBHB12(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['3O', '3O_A', '3O_B', '4O', '4O_A', '4O_B', '2N', '2N_A', '2N_B', '3CL', '3CL_A', '3CL_B']
        self.exp_values = {'2': 2.421, '3': 9.967, '5': 3.021, '10': 3.241}
        self.testset_calculations = {'2': "-1 * ['3O'] +1 * ['3O_A'] +1 * ['3O_B']",
              '3': "-1 * ['4O'] +1 * ['4O_A'] +1 * ['4O_B']",
              '5': "-1 * ['2N'] +1 * ['2N_A'] +1 * ['2N_B']",
              '10': "-1 * ['3CL'] +1 * ['3CL_A'] +1 * ['3CL_B']"}
        self.charges = {'3O': -0.0, '3O_A': 0.0, '3O_B': -0.0, '4O': 0.0, '4O_A': 0.0, '4O_B': 0.0, '2N': 0.0, '2N_A': 0.0, '2N_B': 0.0, '3CL': -0.0, '3CL_A': 0.0, '3CL_B': -0.0}
    