"""
Creates single instances of all test-set classes and returns them as a dictionary 
{name: object}. Modify/import here only when adding a new test-set.
"""
from BH76_full import BH76_full
from W417 import W417
from FracP import FracP
from ACONF import ACONF
from ADIM6 import ADIM6
from AHB21 import AHB21
from ALKBDE10 import ALKBDE10
from AMINO20x4 import AMINO20x4
from BH76 import BH76
from BH76RC import BH76RC
from BHPERI import BHPERI
from BHROT27 import BHROT27
from BSR36 import BSR36
from BUT14DIOL import BUT14DIOL
from CARBHB12 import CARBHB12
from CDIE20 import CDIE20
from CHB6 import CHB6
from DC13 import DC13
from FH51 import FH51
from G21EA import G21EA
from G21IP import G21IP
from G2RC import G2RC
from HAL59 import HAL59
from HEAVY28 import HEAVY28
from ICONF import ICONF
from IL16 import IL16
from INV24 import INV24
from ISO34 import ISO34
from ISOL24 import ISOL24
from MB1643 import MB1643
from MCONF import MCONF
from PArel import PArel
from PCONF21 import PCONF21
from PNICO23 import PNICO23
from RC21 import RC21
from RG18 import RG18
from RSE43 import RSE43
from S66 import S66
from SCONF import SCONF
from SIE4x4 import SIE4x4
from W411 import W411
from WATER27 import WATER27
from YBDE18 import YBDE18
from debugtest import debugtest

def instantiate() -> dict:
    """Returns dictionary {testset_name: class instance}."""
    return {
        'FracP': FracP(),
        'W417': W417(),
        'BH76_full': BH76_full(),
        'ACONF': ACONF(),
        'ADIM6': ADIM6(),
        'AHB21': AHB21(),
        'ALKBDE10': ALKBDE10(),
        'AMINO20x4': AMINO20x4(),
        'BH76': BH76(),
        'BH76RC': BH76RC(),
        'BHPERI': BHPERI(),
        'BHROT27': BHROT27(),
        'BSR36': BSR36(),
        'BUT14DIOL': BUT14DIOL(),
        'CARBHB12': CARBHB12(),
        'CDIE20': CDIE20(),
        'CHB6': CHB6(),
        'DC13': DC13(),
        'FH51': FH51(),
        'G21EA': G21EA(),
        'G21IP': G21IP(),
        'G2RC': G2RC(),
        'HAL59': HAL59(),
        'HEAVY28': HEAVY28(),
        'ICONF': ICONF(),
        'IL16': IL16(),
        'INV24': INV24(),
        'ISO34': ISO34(),
        'ISOL24': ISOL24(),
        'MB1643': MB1643(),
        'MCONF': MCONF(),
        'PArel': PArel(),
        'PCONF21': PCONF21(),
        'PNICO23': PNICO23(),
        'RC21': RC21(),
        'RG18': RG18(),
        'RSE43': RSE43(),
        'S66': S66(),
        'SCONF': SCONF(),
        'SIE4x4': SIE4x4(),
        'W411': W411(),
        'WATER27': WATER27(),
        'YBDE18': YBDE18(),
        'debugtest': debugtest()
    }
