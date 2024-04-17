import json
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
config_pth = current_dir.parent / 'database/mpl/config.json'

def set_mpl_style(mpl, pth=config_pth):
    mpl_custom = json.load(open(str(pth), 'r'))
    for k, v in mpl_custom.items():
        mpl.rcParams[k] = v
    return mpl_custom
