# src/utils/parameters.py

class HYPERPARAMS:
    # ================= PATHS =================
    BASE_MODEL_PATH = "./checkpoints/Qwen3-VL-4B-Instruct"
    INIT_MODEL_PATH = "./checkpoints/Qwen3-VL-GUI-Init"

    # ================= TRAINING PARAMS FOR SFT =================
    # SFT (Semantic injection)
    SFT_SAMPLES_PER_ACTION = 90
    SFT_DATA_PATH = "./data/sft.jsonl"
    SFT_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT"
    
    # Image
    IMAGE_SIZE = 1024
    
    # Optimization
    SFT_LEARN_RATE = 1.5e-5
    SFT_BATCH_SIZE = 4
    SFT_GRAD_ACCUM_STEPS = 4
    SFT_EPOCHS = 1
    SFT_MAX_LENGTH = 4096
    
    # Seeds
    SFT_SEED = 42
    
    # ================= PHASE 2: SFT (ScreenSpot Trajectory) =================
    # Output of Stage 1 is Input of Stage 2
    SFT_2_INPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT" 
    SFT_2_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT-ScreenSpot"
        
    # ScreenSpot Config
    SCREENSPOT_DATA_PATH = "rootsautomation/ScreenSpot"
    SCREENSPOT_TOTAL_SIZE = 1200  # Load 1200 samples total
    SCREENSPOT_TRAIN_RATIO = 0.8  # 80% for Trajectory Training
    SCREENSPOT_EVAL_RATIO = 0.1   # 10% for Eval (Loss calculation)
    # Remaining 10% is reserved as 'Test' (Raw data)
        
    SFT_2_LEARN_RATE = 1e-5 # Slightly lower LR for Stage 2
    SFT_2_BATCH_SIZE = 4    # Images are heavy, smaller batch
    SFT_2_GRAD_ACCUM_STEPS = 4
    SFT_2_EPOCHS = 4        # More epochs to learn visual features
    
    # ================= TRAINING PARAMS FOR GRPO1 =================
    # GRPO1 (Easy grounding tasks with ScreenSpot)
    GRPO1_DATA_PATH = "rootsautomation/ScreenSpot"
    GRPO1_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-ScreenSpot-1"
    
    GRPO1_LEARN_RATE = 5e-6
    GRPO1_BATCH_SIZE = 4
    GRPO1_GRAD_ACCUM_STEPS = 1
    GRPO1_NUM_GENERATIONS = 4
    GRPO1_NUM_EPOCHS = 1  # Number of training epochs
    GRPO1_LOGGING_STEPS = 1  # Log metrics every N steps
    GRPO1_TEMPERATURE = 1.0  # Sampling temperature for generation
    GRPO1_EVAL_STEPS = 50
    GRPO1_SAVE_STEPS = 50
    GRPO1_MAX_CHECKPOINTS = 2
    GRPO_DATASET_SIZE = 600
    
    GRPO1_SEED = 1103
    
    # ================= TRAINING PARAMS FOR GRPO2 =================
    # GRPO2 (Normal navigating tasks with AndroidControl)
    
    # ================= TRAINING PARAMS FOR GRPO3 =================
    # GRPO3 (Hard navigating tasks with GUIOdyssey)
    
    # ================= EVALUATION PARAMS =================
    VLM_API_KEY = "sk-e9d4a0b8248b4802b0458ca26612f25c"
    VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    BASELINE_MODEL_NAME = "qwen3-vl-235b-a22b-instruct" 
    
    # Evaluation Config
    EVAL_DATA_PATH = "rootsautomation/ScreenSpot"
    EVAL_DATASET_SIZE = 10 # Number of samples to test
    EVAL_MAX_STEPS = 20
    EVAL_OUTPUT_DIR = "./eval_results"