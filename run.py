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
GEN_LENGTH_IN_SECONDS = 30

SAMPLING_TEMP = 0.98

UPSAMPLE_LEVEL_0 = True  # High quality upsampling

### --------------- BEGIN NOTEBOOK --------------

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

vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

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
    if os.path.isfile(audio_file):
        print("Found audio file: ", audio_file)
    else:
        print("Could not find file!")
    # Specify how many seconds of audio to prime on.
    prompt_length_in_seconds=PROMPT_LENGTH_IN_SECONDS    
    
### CELL: Resume from last checkpoint file
if os.path.exists(hps.name):
  # Identify the lowest level generated and continue from there.
  for level in [1, 2]:
    data = f"{hps.name}/level_{level}/data.pth.tar"
    if os.path.isfile(data):
        mode = 'upsample'
        codes_file = data
        print('Upsampling from level '+str(level))
        break
print('mode is now '+mode)

### CELL: run cell regardless of which mode you choose
sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

### CELL: Specify your choice of artist, genre, lyrics, and length of musical sample.
sample_length_in_seconds = GEN_LENGTH_IN_SECONDS          
                                       # Full length of musical sample to generate - we find songs in the 1 to 4 minute
                                       # range work well, with generation time proportional to sample length.  
                                       # This total length affects how quickly the model 
                                       # progresses through lyrics (model also generates differently
                                       # depending on if it thinks it's in the beginning, middle, or end of sample)
hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

### CELL: Define Metas
# Note: Metas can contain different prompts per sample.
# By default, all samples use the same prompt.
metas = [dict(artist = "Rick Astley",
            genre = "Pop",
            total_length = hps.sample_length,
            offset = 0,
            lyrics = """
                    Never gonna give you up
                    Never gonna let you down
                    """,
            ),
          ] * hps.n_samples
labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

### CELL: Optionally adjust the sampling temperature
sampling_temperature = SAMPLING_TEMP

lower_batch_size = 16
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 16
lower_level_chunk_size = 32
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size),
                    dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                         chunk_size=lower_level_chunk_size),
                    dict(temp=sampling_temperature, fp16=True, 
                         max_batch_size=max_batch_size, chunk_size=chunk_size)]

### CELL: Sample from the model
if sample_hps.mode == 'ancestral':
    zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(len(priors))]
    zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
elif sample_hps.mode == 'upsample':
    assert sample_hps.codes_file is not None
    # Load codes.
    data = t.load(sample_hps.codes_file, map_location='cpu')
    zs = [z.cuda() for z in data['zs']]
    assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
    del data
    print('Falling through to the upsample step later in the notebook.')
elif sample_hps.mode == 'primed':
    assert sample_hps.audio_file is not None
    audio_files = sample_hps.audio_file.split(',')
    duration = (int(sample_hps.prompt_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
    x = load_prompts(audio_files, duration, hps)
    zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
    zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
else:
    raise ValueError(f'Unknown sample mode {sample_hps.mode}.')
    
if UPSAMPLE_LEVEL_0:
    ### CELL: Load upsamplers
    # Set this False if you are on a local machine that has enough memory (this allows you to do the
    # lyrics alignment visualization during the upsampling stage). For a hosted runtime, 
    # we'll need to go ahead and delete the top_prior if you are using the 5b_lyrics model.
    if False:
      del top_prior
      empty_cache()
      top_prior=None
    upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]
    labels[:2] = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers]

    ### CELL: Upsample!
    zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)