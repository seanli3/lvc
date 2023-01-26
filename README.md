### Installation
* PyTorch
* Pytorch-Geometric
* NetworkX

### Vertex classification
Edit `run/configs.localWL.yaml`, and run
```bash
cd run
python main.py --cfg configs/localWL.yaml --repeat <REPEATS>
```

### Graph classification
Follow ReadMe in `fair_comparison` to prepare datasets, then create a config file similar to `config_LVC.yml`
```bash
python Launch_Experiments --config-file config_LVC.yml --dataset-name <DATASET> --result-folder results --debug
```