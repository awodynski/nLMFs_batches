import sys
sys.path.insert(0, './opt/')
from pathlib import Path

# 3⃣  Reszta importów
from trainer import Trainer
from testset_factory import instantiate as instantiate_testsets
from model_creator import ModelCreator
from tensorflow import keras as K

TESTSET_NAMES      = [ 'FracP',
                       'W417',
                       'BH76_full',
                       'ACONF',
                       'ADIM6',
                       'AHB21',
                       'ALKBDE10',
                       'AMINO20x4',
                       'BH76',
                       'BH76RC',
                       'BHPERI',
                       'BHROT27',
                       'BSR36',
                       'BUT14DIOL',
                       'CARBHB12',
                       'CDIE20',
                       'CHB6',
                       'DC13',
                       'FH51',
                       'G21EA',
                       'G21IP',
                       'G2RC',
                       'HAL59',
                       'HEAVY28',
                       'ICONF',
                       'IL16',
                       'INV24',
                       'ISO34',
                       'ISOL24',
                       'MB1643',
                       'MCONF',
                       'PArel',
                       'PCONF21',
                       'PNICO23',
                       'RC21',
                       'RG18',
                       'RSE43',
                       'S66',
                       'SCONF',
                       'SIE4x4',
                       'UPU23',
                       'W411',
                       'WATER27',
                       'YBDE18'
                       ]

TESTSET_PATHS      = {'FracP':      "/path/to/npy/files/", 
                      'W417':       "/path/to/npy/files/", 
                      'BH76_full':  "/path/to/npy/files/", 
                      'ACONF':      "/path/to/npy/files/", 
                      'ADIM6':      "/path/to/npy/files/", 
                      'AHB21':      "/path/to/npy/files/", 
                      'ALKBDE10':   "/path/to/npy/files/", 
                      'AMINO20x4':  "/path/to/npy/files/", 
                      'BH76':       "/path/to/npy/files/", 
                      'BH76RC':     "/path/to/npy/files/", 
                      'BHPERI':     "/path/to/npy/files/", 
                      'BHROT27':    "/path/to/npy/files/", 
                      'BSR36':      "/path/to/npy/files/", 
                      'BUT14DIOL':  "/path/to/npy/files/", 
                      'CARBHB12':   "/path/to/npy/files/", 
                      'CDIE20':     "/path/to/npy/files/", 
                      'CHB6':       "/path/to/npy/files/", 
                      'DC13':       "/path/to/npy/files/", 
                      'FH51':       "/path/to/npy/files/", 
                      'G21EA':      "/path/to/npy/files/", 
                      'G21IP':      "/path/to/npy/files/", 
                      'G2RC':       "/path/to/npy/files/", 
                      'HAL59':      "/path/to/npy/files/", 
                      'HEAVY28':    "/path/to/npy/files/", 
                      'ICONF':      "/path/to/npy/files/", 
                      'IL16':       "/path/to/npy/files/", 
                      'INV24':      "/path/to/npy/files/", 
                      'ISO34':      "/path/to/npy/files/", 
                      'ISOL24':     "/path/to/npy/files/", 
                      'MB1643':     "/path/to/npy/files/", 
                      'MCONF':      "/path/to/npy/files/", 
                      'PArel':      "/path/to/npy/files/", 
                      'PCONF21':    "/path/to/npy/files/", 
                      'PNICO23':    "/path/to/npy/files/", 
                      'RC21':       "/path/to/npy/files/", 
                      'RG18':       "/path/to/npy/files/", 
                      'RSE43':      "/path/to/npy/files/", 
                      'S66':        "/path/to/npy/files/", 
                      'SCONF':      "/path/to/npy/files/", 
                      'SIE4x4':     "/path/to/npy/files/", 
                      'UPU23':      "/path/to/npy/files/", 
                      'W411':       "/path/to/npy/files/", 
                      'WATER27':    "/path/to/npy/files/", 
                      'YBDE18':     "/path/to/npy/files/"}


SCALE = {
         'FracP'    :       10.00,
         'W417'     :       50.00,
         'BH76_full':      100.00,
         'ACONF'    :       92.97,
         'ADIM6'    :       16.93,
         'AHB21'    :        5.06,
         'ALKBDE10' :        0.56,
         'AMINO20x4':       69.93,
         'BH76'     :       12.20,
         'BH76RC'   :        2.66,
         'BHPERI'   :       10.88,
         'BHROT27'  :       27.18,
         'BSR36'    :        7.02,
         'BUT14DIOL':       60.90,
         'CARBHB12' :       37.68,
         'CDIE20'   :       28.04,
         'CHB6'     :        2.12,
         'DC13'     :        2.06,
         'FH51'     :        1.83,
         'G21EA'    :        5.07,
         'G21IP'    :        0.22,
         'G2RC'     :        1.11,
         'HAL59'    :       74.28,
         'HEAVY28'  :       91.58,
         'ICONF'    :       17.40,
         'IL16'     :        0.52,
         'INV24'    :        5.34,
         'ISO34'    :        7.80,
         'ISOL24'   :        2.59,
         'MB1643'   :        0.24,
         'MCONF'    :       45.72,
         'PArel'    :       49.12,
         'PCONF21'  :      105.15,
         'PNICO23'  :       26.60,
         'RC21'     :        1.59,
         'RG18'     :      196.00,
         'RSE43'    :       22.44,
         'S66'      :       52.00,
         'SCONF'    :       12.36,
         'SIE4x4'   :        3.38,
         'UPU23'    :        9.93,
         'W411'     :        1.71,
         'WATER27'  :        1.40,
         'YBDE18'   :        1.15,
         }

def main() -> None:
    # ------------------------------------------------------------------
    # TRAINING HYPER-PARAMETERS
    # ------------------------------------------------------------------
    NUM_INPUTS        = 9
    FEATURES          = [True, True, True, True, False, False]
    EPOCHS            = 1000
    LEARNING_RATE     = 1.0e-3
    L2_RATIO          = 0.0
    HIDDEN_UNITS      = 128
    NUM_LAYERS        = 4
    ACTIVATION_HIDDEN = 'gelu'
    ACTIVATION_OUTPUT = 'sigmoid'
    INPUT_SQUEEZE     = 'SignedLogTransform'
    LINEAR_SCALING    = 'm1p1'
    NEG_PUNISH        = 1000.0

    # Dispersion parameters (a1, a2, s8)
    A1 = 0.4622487557716163
    A2 = 3.6751015046553417
    S6 = 1.0
    S8 = 0.16091066325898445
    
    MP2      = False

    # Model exchange (only PBE available)
    X_MODEL  = 'PBE'

    # Model correlation (B97/B95 model available)
    C_MODEL  = 'B97'
    # Linear parameters for B97/B95 correlation model
    SCAL_OPP = [1.40138769, -2.90212804, 3.41403089, -1.83609439, 0.0, 0.0]
    SCAL_SS  = [0.22875368, -0.52818435, 0.91060473, -0.6228437, 0.0, 0.0, 0.0]
    # Nonlinear parameters for B97/B95 correlation model
    C_SS     = 0.09544
    C_OPP    = 0.004987

    # Range separtation
    RS_MODEL = False

    # Strong correlation model (available Pade and Erf model) QAC_MODEL=['none'] gives no strong correlation
    QAC_MODEL = ['Pade', 0.089933, 0.134653, 0.725653]

    # Restart weights
    RESTART_FILE=None

    # Create model
    creator = ModelCreator(
        num_inputs                  = NUM_INPUTS,
        input_squeeze               = INPUT_SQUEEZE,
        number_of_layers            = NUM_LAYERS,
        hidden_units                = HIDDEN_UNITS,
        activation_function_hidden  = ACTIVATION_HIDDEN,
        activation_function_output  = ACTIVATION_OUTPUT,
        linear_scaling              = LINEAR_SCALING,
        l2_ratio                    = L2_RATIO,
        x_model                     = X_MODEL,
        c_model                     = C_MODEL,
        nlx                         = 1.0, #we are always using full PBEx
        scal_opp                    = SCAL_OPP,
        scal_ss                     = SCAL_SS,
        seed                        = 42,
        c_ss                        = C_SS,
        c_opp                       = C_OPP,
        corr_train                  = False,
        rs_model                    = RS_MODEL,
        qac                         = QAC_MODEL,
        restart                     = RESTART_FILE
    )

    model   = creator.create_model()
    optim   = K.optimizers.Adam(learning_rate=LEARNING_RATE)
    trainer = Trainer(
            model           = model, 
            optimizer       = optim, 
            testset_objects = instantiate_testsets(), 
            s6              = S6, 
            s8              = S8, 
            a1              = A1, 
            a2              = A2,
            testset_names   = TESTSET_NAMES, 
            testset_paths   = TESTSET_PATHS, 
            scale           = SCALE,
            features        = FEATURES, 
            neg_punish      = NEG_PUNISH,
            epochs          = EPOCHS)

    trainer.train(checkpoint_dir=Path("."))


if __name__ == "__main__":
    main()
