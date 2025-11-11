"""
Configuration file for PyBanglaCodeActAgent.

Modify these settings to customize the agent behavior.
"""

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_MODEL_LEN = 4096
ENABLE_PREFIX_CACHING = True
# QUANTIZATION = "awq"  # Uncomment to enable quantization

# Agent Configuration
MAX_ITERATIONS = 4
MAX_RETRIES = 15
TIMEOUT_SECONDS = 5

# Sampling Parameters
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 4096
REPETITION_PENALTY = 1.05

# Self-Consistency Settings
SELF_CONSISTENCY_PATHS = 16
SELF_CONSISTENCY_ENABLED = False

# File Paths
DEFAULT_INPUT_CSV = "dev.csv"
DEFAULT_OUTPUT_JSON = "submission.json"

# Random Seed
RANDOM_SEED = 42

# Logging
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
ENABLE_COLOR_OUTPUT = True
ENABLE_SYNTAX_HIGHLIGHTING = True
