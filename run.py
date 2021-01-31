### --------------- IMPORTS -------------------
import jukebox
import torch as t
import librosa
import os
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample, \
                           load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
rank, local_rank, device = setup_dist_from_mpi()

### --------------- GLOBAL CONFIGS --------------
INPUT_DIR = "./inputs"
OUTPUT_DIR = "./outputs"

MODE = 'primed'
PROMPT_LENGTH_IN_SECONDS = 12

### CELL: Sample from the 5B or 1B Lyrics Model
model = '5b_lyrics' # or '5b' or '1b_lyrics'
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model in ('5b', '5b_lyrics') else 8
# Specifies the directory to save the sample in.
hps.name = OUTPUT_DIR
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 16
hps.levels = 3
hps.hop_fraction = [.5,.5,.125]

#vqvae, *priors = MODELS[model]
#vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), device)
#top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

### CELL: Select Mode
if MODE == 'ancestral':
    # The default mode of operation.
    mode = 'ancestral'
    codes_file=None
    audio_file=None
    prompt_length_in_seconds=None
elif MODE == 'primed':
    # Prime song creation using an arbitrary audio sample.
    mode = 'primed'
    codes_file=None
    # Specify an audio file here.
    audio_file = './inputs/primer_test.wav'
    if os.path.isfile():
        print("Using audio file: ", Specify)
    # Specify how many seconds of audio to prime on.
    prompt_length_in_seconds=PROMPT_LENGTH_IN_SECONDS