import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_t2v_A14B import t2v_A14B
from .wan_t2v_1_3B import t2v_1_3B

WAN_CONFIGS = {
    't2v-A14B': t2v_A14B,
    't2v-1.3B': t2v_1_3B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
}

SUPPORTED_SIZES = {
    't2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    't2v-1.3B': ('720*1280', '1280*720', '480*832', '832*480'),
}
