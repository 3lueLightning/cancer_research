import glob
import os

import constants

def load_last_tuning():
    list_of_files = glob.glob( constants.CANCER_DIRECTORY + INTERPRETABLE_DIR
                              + PROTEIN_SET + 'xgb_models/*.pkl')
    latest_file = max(list_of_files, key=os.path.getctime)
    fn = latest_file 

    tuning = joblib.load(fn)
    print(f'number of tuning steps: {len(tuning["fs_scores"])}')

    return tuning