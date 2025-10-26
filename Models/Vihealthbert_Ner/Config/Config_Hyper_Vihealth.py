import os
MODEL_NAME ="demdecuong/vihealthbert-base-word"
PROJECT_ROOT=os.path.abspath( os.path.join(os.path.dirname(__file__),"..\\..\\..\\"))
DATA_DIR = os.path.join(PROJECT_ROOT, "Datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Model_save", "Vihealthbert_Ner")

MAX_LEN = 200
BATCH_SIZE = 16
NUM_EPOCHS = 40               
LR = 2e-5                   
WEIGHT_DECAY = 0.01           
WARMUP_RATIO = 0.06            
GRAD_ACCUM_STEPS = 2         
NUM_WORKERS = 8               
FP16 = True                   
MAX_GRAD_NORM = 1.0
SAVE_TOTAL_LIMIT = 1          
LOGGING_STEPS = 20
METRIC_FOR_BEST_MODEL = "f1"
LABEL_SMOOTHING = 0.05
LR_SCHEDULER_TYPE = "cosine"
OPTIMIZER = "adamw_torch"
WARMUP_STEPS = 0
EARLY_STOPPING_PATIENCE = 20
SEED = 42   
