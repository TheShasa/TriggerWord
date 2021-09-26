
NFTT = 200  # Length of each window segment
FS = 8000  # Sampling frequencies
N_OVERLAP = 120  # Overlap between windows


TX = 5511  # The number of time steps input to the model from the spectrogram
N_FREQ = 101  # Number of frequencies input to the model at each time step of the spectrogram

TY = 1375  # The number of time steps in the output of our model

ACTIVE_FOLDER = 'raw_data/blyad_sliced/'
BACKGROUND_FOLDER = 'raw_data/backgrounds_office/'
NEGATIVE_FOLDER = 'raw_data/norm_sliced/'

TRAIN_SIZE = 4000
