from testsetclass import TestsetClass


class BHPERI(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['Cyclobutene', 'TS1', '13r_7', '13_c2h4', '13ts_7a', '05r', '00r', '06ts', '09r', '09ts']
        self.exp_values = {'0': 35.3, '16': 13.1, '22': 27.8, '25': 31.3}
        self.testset_calculations = {'0': "-1 * ['Cyclobutene'] +1 * ['TS1']",
              '16': "-1 * ['13r_7'] -1 * ['13_c2h4'] +1 * ['13ts_7a']",
              '22': "-1 * ['05r'] -1 * ['00r'] +1 * ['06ts']",
              '25': "-1 * ['09r'] -1 * ['00r'] +1 * ['09ts']"}
        self.charges = {'Cyclobutene': 0.0, 'TS1': 0.0, '13r_7': 0.0, '13_c2h4': 0.0, '13ts_7a': -0.0, '05r': 0.0, '00r': 0.0, '06ts': -0.0, '09r': 0.0, '09ts': 0.0}
    