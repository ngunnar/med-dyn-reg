# NOTE: THIS DESCRIPTION IS UNDER CONSTRUCTION

# Unsupervised dynamic modeling of medical image transformations
Official implementation of **Unsupervised dynamic modeling of medical
image transformations** accepted at 25th International Conference on Information Fusion 2022 (https://www.fusion2022.se/).

[[Paper]](https://ieeexplore.ieee.org/abstract/document/9841369) [[Presentation]](https://drive.google.com/file/d/10FoyxB1BT0c3Ej_vSuMIPv5-G0mtKmI-/view?usp=sharing) [[Dataset]](https://echonet.github.io/dynamic/)

## Enviroment
We use the following libraries

```
tensorflow 2.4.1
tensorflow-probability 0.12.1
numpy 1.19.5
cv2 4.5.1
pandas 1.2.2
tqdm 4.57.0
json 2.0.9
voxelmorph 0.1
SimpleITK 2.0.2
tqdm 4.56.0
matplotlib 3.3.4
```

## Architecture
We model the motion by mapping high-dimensional sequential images to a low-dimensional space wherein a linear relationship between a hidden state process and the lower-dimensional representation holds. For this, we use a Conditional Variational Auto-Encoder (CVAE) to nonlinearly map the higher-dimensional images to a lower-dimensional space. In this lower-dimensional space, we model the dynamics with a Linear Gaussian State Space Model (LGSSM). Using the decoder we upsample the dynamics, from the low-dimensional space to the image space and estimate the spatial transformation between a reference image and the current. The image reconstruction can the be obtain by transform the reference image with the spatial transformation.

During training the CVAE - and LGSSM parameters are jointly updated by maximizing the evidence lower bound of the log likelihood:

<p align="center">
<img src="https://user-images.githubusercontent.com/10964648/192537200-0b74a63e-231b-4a1b-bf50-d1eee847fcb5.png" alt="Architecture" width="45%"/>
</p>

In the current implementation we use a image sequence of $50$ images $\mathbf{y} = [y_1, \dots, y_{50}]$ where $y_i \in R^{112 \times 112}$. The images was reduced using the encoder such that $x \in R^{16}$ and for state space variable $z$ we used $z \in R^{32}$.

<figure>
<p align="center">
  <img src="https://user-images.githubusercontent.com/10964648/192531044-3ecb7c4c-5466-45e2-853f-82af7edb8e31.png" alt="Architecture" width="45%"/>
  <img src="https://user-images.githubusercontent.com/10964648/192531267-a8239035-a409-481d-a2fa-9ad77b1e894e.png" alt="Architecture" width="45%"/>
</p>
<figcaption align = "center"><b>Fig.1 - Illustration and graphical representation of the model.</b></figcaption>
</figure>



## Run (TODO)
To train the model run
```
python train.py --dim_z 32 --gpus 1 --prefix Test --ds_path PATH_TO_DS
```
Following arguments is supported
```
--help            show this help message and exit
--dim_y           dimension of image variable (default (112,112))
--dim_x           dimension of latent variable (default 16)
--dim_z           dimension of state space variable (default 32)
--dec_in          input dim to decoder (default 16)
--saved_model     model path if continue running model (default:None)
--start_epoch     start epoch
--gpus            comma separated list of GPUs (default -1 (CPU))
--prefix          predix for log folder (default:None)
--ds_path         path to dataset (default:/data/Niklas/EchoNet-Dynamics)
--ds_size         Size of datasets
```

To test
```
import json
import numpy as np
from src.datasetLoader import TensorflowDatasetLoader

def get_config(path):
    with open('%s/%s' % (os.path.dirname(path), 'config.json')) as data_file:
        config_dict = json.load(data_file)
    config = namedtuple("Config", config_dict.keys())(*config_dict.values())
    return config, config_dict
    
path = './logs/kvae/32_kvae/32_Test/'
kvae_dir = path
kvae_config, kvae_config_dict = get_config(kvae_dir)
kvae_model = KVAE(config = kvae_config)
kvae_model.load_weights(kvae_dir + 'kvae_cardiac_32').expect_partial()

# Load validation data
val_generator = TensorflowDatasetLoader(root = PATH_TO_DS, image_shape = tuple(kvae_config.dim_y), length = kvae_config.ph_steps, split='val', period = kvae_config.period)

# Reconstruction 
val_batch = list(val_generator.data.take(1).as_numpy_iterator())[0]
inputs = [val_batch[0][None,...], val_batch[1][None,...]]
y_filt_val, y_smooth_val, y_vae_val = kvae_model.predict(inputs)

# Impute/predict
mask = np.ones(shape=50).astype('bool')
known = np.arange(50)
steps = 5
last = 1
known = known[:-last][::steps]
mask[known] = False
inputs = [val_batch[0][None,...], mask[None,...]]
y_filt_val, y_smooth_val, y_vae_val = kvae_model.predict(inputs)

# Latent space analysis
x_mu_smooth, x_cov_smooth, x_mu_filt, x_covs_filt, x_mu_filt_pred, x_covs_filt_pred = kvae_model.get_latents(inputs)
```

## Result (TODO)
### Smooth, filter and one-step-ahead prediction when all frames are known

https://user-images.githubusercontent.com/10964648/192530051-a5af79c5-7123-4052-8dda-09fecab2f952.mp4

### Smooth, filter and prediction when every 3rd frames are known

https://user-images.githubusercontent.com/10964648/192530759-11a926e9-73a3-4aa3-a603-5e014ea55d31.mp4


## Citation
```
@INPROCEEDINGS{9841369,
  author={Gunnarsson, Niklas and Sjölund, Jens and Kimstrand, Peter and Schön, Thomas B.},
  booktitle={2022 25th International Conference on Information Fusion (FUSION)}, 
  title={Unsupervised dynamic modeling of medical image transformations}, 
  year={2022},
  volume={},
  number={},
  pages={01-07},
  doi={10.23919/FUSION49751.2022.9841369}}
```
