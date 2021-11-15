# Latent linear dynamics in spatiotemporal medical data
Official implementation of Latent linear dynamics in spatiotemporal medical data ([Arxiv](https://arxiv.org/abs/2103.00930)).

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

## Architecture (TODO)
<img src="https://user-images.githubusercontent.com/10964648/141752830-dcfaa9ba-d009-400f-8c22-895fad8da666.PNG" alt="Architecture" width="350"/>

<img src="https://user-images.githubusercontent.com/10964648/109638288-239ba300-7b4e-11eb-9450-4e1e60f2a86c.PNG" alt="Architecture" width="350"/>

The intensity dynamics is modelled with a Linear Gaussian State Space Model (LGSSM), not on the raw image time series but on the latent process generated by a Variational Auto-Encoder (VAE). The VAE - and LGSSM parameters are jointly updated by maximizing the evidence lower bound of the log likelihood:

<img src="https://render.githubusercontent.com/render/math?math=\log p_\theta(\mathbf{y} \mid y_M) \geq \mathbb{E}_{q_{\phi}(\mathbf{x}, x_M \mid \mathbf{y}, y_M)}          \Big[\log \dfrac{p_\theta(\mathbf{y}\mid \mathbf{x}, y_M)}{q_\phi(\mathbf{x}, x_M\mid \mathbf{y}, y_M)} + \mathbb{E}_{p_\gamma(\mathbf{z}\mid \mathbf{x}, x_M)}\Big[\dfrac{p_\gamma(\mathbf{x}, \mathbf{z} \mid x_M)}{p_\gamma(\mathbf{z} \mid \mathbf{x}, x_M)}\Big]\Big].">

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

## Result
### Smooth and filter imputing when every 5th frame is known
![impute5](https://user-images.githubusercontent.com/10964648/109649372-1dacbe80-7b5c-11eb-978f-09ff9ae47c05.gif)

### Filter prediction when last 5th frames are unknown
![predict5](https://user-images.githubusercontent.com/10964648/109648801-55ffcd00-7b5b-11eb-9f0e-dc559ebc6d5d.gif)

## Citation
```
@misc{gunnarsson2021latent,
      title={Latent linear dynamics in spatiotemporal medical data}, 
      author={Niklas Gunnarsson and Jens Sjölund and Thomas B. Schön},
      year={2021},
      eprint={2103.00930},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph}
}
```
