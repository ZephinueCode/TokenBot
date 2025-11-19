# src/utils/parameters.py

class HYPERPARAMS:
    # ================= PATHS =================
    BASE_MODEL_PATH = "./checkpoints/Qwen3-VL-4B-Instruct"
    INIT_MODEL_PATH = "./checkpoints/Qwen3-VL-GUI-Init"

    # ================= TRAINING PARAMS FOR SFT =================
    # SFT (Semantic injection)
    SFT_SAMPLES_PER_ACTION = 50
    SFT_DATA_PATH = "./data/sft.jsonl"
    SFT_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT"
    
    # Image
    IMAGE_SIZE = 640
    
    # Optimization
    SFT_LEARN_RATE = 1.8e-5
    SFT_BATCH_SIZE = 4
    SFT_GRAD_ACCUM_STEPS = 4
    SFT_EPOCHS = 1
    SFT_MAX_LENGTH = 4096
    
    # Seeds
    SFT_SEED = 42
    
    # ================= TRAINING PARAMS FOR GRPO1 =================
    # GRPO1 (Easy grounding tasks with ScreenSpot)
    GRPO1_DATA_PATH = "rootsautomation/ScreenSpot"
    GRPO1_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-Grounding"
    
    GRPO1_LEARN_RATE = 5e-6
    GRPO1_BATCH_SIZE = 16
    GRPO1_GRAD_ACCUM_STEPS = 1
    GRPO1_NUM_GENERATIONS = 8
    GRPO1_NUM_EPOCHS = 1  # Number of training epochs
    GRPO1_LOGGING_STEPS = 1  # Log metrics every N steps
    GRPO1_TEMPERATURE = 1.0  # Sampling temperature for generation
    GRPO1_WARMUP_STEPS = 100  # Warmup steps for learning rate scheduler
    GRPO1_EVAL_STEPS = 100
    GRPO1_SAVE_STEPS = 100
    GRPO1_MAX_CHECKPOINTS = 2
    
    GRPO1_SEED = 42
    
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
    EVAL_DATASET_SIZE = 1 # Number of samples to test
    EVAL_MAX_STEPS = 100
    EVAL_OUTPUT_DIR = "./eval_results"