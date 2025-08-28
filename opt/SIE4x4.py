from testsetclass import TestsetClass


class SIE4x4(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['he2+_1.5', 'he', 'he+', 'nh3', 'nh3+', 'nh32+_1.25']
        self.exp_values = {'6': 31.3,  '9': 25.9}
        self.testset_calculations = {'6': "+1 * ['he'] +1 * ['he+'] -1 * ['he2+_1.5']",
            '9': "+1 * ['nh3'] +1 * ['nh3+'] -1 * ['nh32+_1.25']"}
        self.charges = {'he2+_1.5': 1.0, 'he': 0.0, 'he+': 1.0, 'nh3': 0.0, 'nh3+': 1.0, 'nh32+_1.25': 1.0}
    