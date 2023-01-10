# tractolearn

Tractography learning.

## Patent

J. H. Legarreta, M. Descoteaux, and P.-M. Jodoin. “PROCESSING OF TRACTOGRAPHY
RESULTS USING AN AUTOENCODER”. Filed 03 2021. Imeka Solutions Inc. United States
Patent #17/337,413. Pending.

## Installation

To use tractolearn, it is recommended to create a virtual environment using python 3.8 that
will host the necessary dependencies:

```sh
   virtualenv tractolearn_env --python=python3.8
   source tractolearn_env/bin/activate
```

They can be installed by executing:

```sh
   pip install -r requirements.txt
   pip install -r requirements_dmri.txt
```

Torch not included. Tested with an NVIDIA RTX 3090 with:

```sh
   pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Once the dependencies installed, `tractolearn` can be installed from its sources
by executing, at its root:

```sh
   pip install -e .
```

In order to execute experiments reporting to [Comet](https://www.comet.ml/site/), an `api_key` needs to be set as an environment variable named `COMETML`. You can write this command in you `.bashrc`

```sh
   export COMETML="api_key"
```

## Training models

To train deep learning models, you need to launch the script [ae_train.py](scripts/ae_train.py). This script takes a config file with all training parameters such as epochs, datasets path, etc. The most up-to-date config file is [config.yaml](configs/train_config.yaml). You can launch the training pipeline with the following command:
```sh
   ae_train.py train_config.yaml -vv
```

## Contributing

## References

## License

This software is distributed under a particular license. Please see the
[*LICENSE*](LICENSE) file for details.
