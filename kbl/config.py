# settings shared by record + test mode
# if you change these you need to re-record (or at least retrain)

SR = 44100
DATA_DIR = "data/samples"
MODEL_PATH = "models/model.joblib"

HOP = 88            # ~2ms at 44.1k, used for the loudness envelope
HIGHPASS_HZ = 150   # cuts out desk thumps / hum

# saved clips are 30ms before the click -> 200ms after
# the extra room at the start lets training shift clips around a bit
SAVE_PRE = 0.030
SAVE_POST = 0.200

# part of the clip we actually compute features on
CLIP_PRE = 0.005
CLIP_LEN = 0.150

# record mode: key events and audio don't line up perfectly so search around the event
SEARCH_BEFORE = 0.12
SEARCH_AFTER = 0.10
MIN_GAP = 0.20      # skip presses closer than this, the sounds overlap

# test mode
LIVE_BEFORE = 0.02
LIVE_AFTER = 0.08
REFRACTORY = 0.08   # don't retrigger right after a click

RELEASE_LABEL = "_release"
