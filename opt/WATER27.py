from testsetclass import TestsetClass


class WATER27(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['H2O', 'H3Op', 'H3OpH2O3', 'OHmH2O', 'OHm']
        self.exp_values = {'16': 76.755,'19': 26.687}
        self.testset_calculations = {'16': "-1 * ['H3OpH2O3'] +1 * ['H3Op'] +3 * ['H2O']",
            '19': "-1 * ['OHmH2O'] +1 * ['OHm'] +1 * ['H2O']"}
        self.charges = {'H2O': 0.0, 'H3Op': 1.0, 'H3OpH2O3': 1.0, 'OHmH2O': -1.0, 'OHm': -1.0}
    