import os

CANCER_DIRECTORY = "/home/jupyter/cancer_research"
DATA_DIRECTORY = os.path.join(CANCER_DIRECTORY, 'data')
MOCK_DATA_DIRECTORY = os.path.join(CANCER_DIRECTORY, 'mock_data')
RESULTS_DIRECTORY = os.path.join(CANCER_DIRECTORY, 'results')

NA_VALUE = 0.0099

# for NN model / check if still usefull
DIRECTORY = "/home/jupyter/cancer_model/agg/" # '/content/drive/My Drive/cancer_study/agg/'
LOGS = DIRECTORY + 'logs'
MODELS = DIRECTORY + 'models'
MAIN_METRIC = 'categorical_accuracy'
PERFORMANCE_FILE = 'nn_performance.pkl'

ENERGY_PROTEINS = ['CAT', 'FBP1', 'FBP2', 'GCLC', 'GCLM', 'GGT1', 'GGT6', 'GSR',
       'GSS', 'GSTA1', 'GSTA2', 'GSTK1', 'GSTM1', 'GSTM2', 'GSTM3',
       'GSTO1', 'GSTP1', 'GSTT1', 'GSTZ1', 'MGST1', 'MGST2', 'MGST3',
       'SDHA', 'SDHB', 'SOD1', 'SOD2', 'SOD3', 'SRC']

EQUATION_SIMPLIFIER = {
    "TRAP1 + HSP90AA1 + HSP90AB1 + HSPB1 + HSP90B1": "HSP90 + HSPB1",
    "HSPA2 + HSPA6 + HSPA8 + HSPA5 + HSPA9 + HSPA12A": "HSP70",
    "DNAJA1 + DNAJA2 + DNAJC11 + DNAJB1 + DNAJC5 + DNAJC13": "DNAJ",
    "HSP90AA1 + HSP90AB1 + HSP90B1 + TRAP1": "HSP90"
}