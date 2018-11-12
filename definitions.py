import os

ROOT_DIR = '/Users/Jan/LRZBOX/Master/LR_Kuehn/'
# ROOT_DIR = '/home/boelts/Dropbox/Master/LR_Kuehn/'

DATA_PATH = os.path.join(ROOT_DIR, 'data')

# paths to raw data
DATA_PATH_REST = os.path.join(ROOT_DIR, 'data', 'dystonia_rest', 'for_python')

DATA_PATH_BAROW = os.path.join(ROOT_DIR, 'data', 'dystonia_stim', 'for_python')

# paths to analyzed data
SAVE_PATH_DATA = os.path.join(ROOT_DIR, 'analysis_data')

SAVE_PATH_DATA_REST = os.path.join(ROOT_DIR, 'analysis_data', 'rest_data')

SAVE_PATH_DATA_BAROW = os.path.join(ROOT_DIR, 'analysis_data', 'stimulation_data')

# paths to figures
SAVE_PATH_FIGURES = os.path.join(ROOT_DIR, 'analysis_figures')
SAVE_PATH_FIGURES_REST = os.path.join(ROOT_DIR, 'analysis_figures', 'rest_data')

SAVE_PATH_FIGURES_BAROW = os.path.join(ROOT_DIR, 'analysis_figures', 'stimulation_data')
