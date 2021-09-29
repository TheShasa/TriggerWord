import numpy as np
from utils.td_utils import load_raw_audio
import utils.config as config
import utils.general as general
from tqdm import tqdm

# Load audio segments using pydub
activates, negatives, backgrounds = load_raw_audio(
    config.BACKGROUND_FOLDER, config.ACTIVE_FOLDER, config.NEGATIVE_FOLDER)

# Should be 10,000, since it is a 10 sec clip
print("background len: " + str(len(backgrounds[0])))
print("activate[0] len: " + str(len(activates[0])))
# Different "activate" clips can have different lengths
print("activate[1] len: " + str(len(activates[1])))
print("negative[0] len: " + str(len(negatives[0])))
# Different "activate" clips can have different lengths
print("negative[1] len: " + str(len(negatives[1])))

print('Number of backgrounds : ' + str(len(backgrounds)))

back_index = np.random.randint(
    low=0, high=len(backgrounds), size=config.TRAIN_SIZE)
print(back_index.shape)
print(back_index[:10])

for path in ['Data/X_train/', 'Data/Y_train/']:
    if not os.path.exists(path):
        os.makedirs(path)
for i in tqdm(range(config.TRAIN_SIZE)):
    x, y = general.create_training_example(
        backgrounds[back_index[i]], activates, negatives)
    np.save('Data/X_train/'+str(i)+'.npy', x)
    np.save('Data/Y_train/'+str(i)+'.npy', y)
