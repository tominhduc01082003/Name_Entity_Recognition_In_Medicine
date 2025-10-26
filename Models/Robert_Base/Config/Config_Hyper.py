import os

MODEL_NAME = "xlm-roberta-base"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "Datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Model_save", "Robert_Base")

MAX_LEN = 200
BATCH_SIZE = 16
NUM_EPOCHS = 15                
LR = 3e-5                     
WEIGHT_DECAY = 0.01           
WARMUP_RATIO = 0.1            
GRAD_ACCUM_STEPS = 1         
NUM_WORKERS = 8               
FP16 = True                   
MAX_GRAD_NORM = 1.0
SAVE_TOTAL_LIMIT = 1          
LOGGING_STEPS = 50
METRIC_FOR_BEST_MODEL = "f1"