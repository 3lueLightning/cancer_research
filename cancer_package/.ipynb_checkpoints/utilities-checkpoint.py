import os
import re
import glob

from sklearn.utils import class_weight

from . import constants

def load_last_tuning():
    list_of_files = glob.glob( constants.CANCER_DIRECTORY + INTERPRETABLE_DIR
                              + PROTEIN_SET + 'xgb_models/*.pkl')
    latest_file = max(list_of_files, key=os.path.getctime)
    fn = latest_file 

    tuning = joblib.load(fn)
    print(f'number of tuning steps: {len(tuning["fs_scores"])}')

    return tuning


def multiple_replace(original_str: str, replacement_dict):
    new_str = original_str
    for old_substr, new_substr in replacement_dict.items():
        new_str = new_str.replace(old_substr, new_substr)
    if new_str == original_str:
        return original_str
    else:
        single_elems_with_parenthesis = re.findall("\([a-zA-Z0-9]*\)", new_str)
        if single_elems_with_parenthesis:
            for elem in single_elems_with_parenthesis:
                new_str = re.sub("\(" + elem[1:-1] + "\)", elem[1:-1], new_str)
            return new_str      
        else:
            return new_str

            
def class_weights(y):
    y_classes = y.unique()
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=y_classes,
        y=y.values)
    class_weights = {key: val for key, val in zip(y_classes, class_weights)}
    return y.map(class_weights)
            