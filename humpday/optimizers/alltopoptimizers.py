# So horrendous this needs a rewrite

from humpday.optimizers.nevergradcube import NEVERGRAD_TOP_OPTIMIZERS
from humpday.optimizers.optunacube import OPTUNA_TOP_OPTIMIZERS
from humpday.optimizers.nloptcube import NLOPT_TOP_OPTIMIZERS
from humpday.optimizers.dlibcube import DLIB_TOP_OPTIMIZERS
from humpday.optimizers.pysotcube import PYSOT_TOP_OPTIMIZERS
from humpday.optimizers.pymoocube import PYMOO_TOP_OPTIMIZERS
from humpday.optimizers.freelunchcube import FREELUNCH_TOP_OPTIMIZERS
from humpday.optimizers.bobyqacube import BOBYQA_TOP_OPTIMIZERS
from humpday.optimizers.axcube import AX_TOP_OPTIMIZERS
from humpday.optimizers.scipycube import SCIPY_TOP_OPTIMIZERS

# free lunch broken
TOP_OPTIMIZERS = NEVERGRAD_TOP_OPTIMIZERS + OPTUNA_TOP_OPTIMIZERS + NLOPT_TOP_OPTIMIZERS\
                 + DLIB_TOP_OPTIMIZERS + PYSOT_TOP_OPTIMIZERS + PYMOO_TOP_OPTIMIZERS\
                  + BOBYQA_TOP_OPTIMIZERS + AX_TOP_OPTIMIZERS+ SCIPY_TOP_OPTIMIZERS






