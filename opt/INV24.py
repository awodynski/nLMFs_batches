from testsetclass import TestsetClass


class INV24(TestsetClass):
    def __init__(self):
        self.weak = 1.0
        self.molecules = ['Dibenzocycloheptene', 'Dibenzocycloheptene_TS', 'BN_Corannulene', 'BN_Corannulene_TS', 'BN_Sumanene', 'BN_Sumanene_TS']
        self.exp_values = {'11': 10.3, '18': 6.2, '21': 27.2}
        self.testset_calculations = {'11': "-1 * ['Dibenzocycloheptene'] +1 * ['Dibenzocycloheptene_TS']",
            '18': "-1 * ['BN_Corannulene'] +1 * ['BN_Corannulene_TS']",
            '21': "-1 * ['BN_Sumanene'] +1 * ['BN_Sumanene_TS']"}
        self.charges = {'Dibenzocycloheptene': 0.0, 'Dibenzocycloheptene_TS': 0.0, 'BN_Corannulene': 0.0, 'BN_Corannulene_TS': -0.0, 'BN_Sumanene': -0.0, 'BN_Sumanene_TS': 0.0}
    