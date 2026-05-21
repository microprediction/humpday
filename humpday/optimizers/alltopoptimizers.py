# Simplified top optimizers focusing on working packages only

from humpday.optimizers.optunacube import OPTUNA_OPTIMIZERS
from humpday.optimizers.scipycube import SCIPY_OPTIMIZERS
from humpday.optimizers.primacube import PRIMA_OPTIMIZERS

# Use the working optimizers as top optimizers
# Take a subset of the best performing ones
try:
    TOP_OPTIMIZERS = SCIPY_OPTIMIZERS[:3] + PRIMA_OPTIMIZERS[:2] + OPTUNA_OPTIMIZERS[:3]
except (ImportError, AttributeError):
    # Fallback if lists don't exist
    TOP_OPTIMIZERS = []






