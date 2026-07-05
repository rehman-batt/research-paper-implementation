import yaml
import argparse
from pathlib import Path
from models.alexnet.model import AlexNet
import torch
import torch.nn as nn

MODELS = {
    "alexnet": AlexNet
}

def arguments() -> argparse.Namespace:
   
    parser = argparse.ArgumentParser() 
    parser.add_argument("config")
    args = parser.parse_args()
    return args

    

def read_config_params(config_path: str) -> dict:
    if not Path(config_path).exists():
        raise FileExistsError("Yaml File Does Not Exist")
    
    with open(config_path) as f:
        params = yaml.safe_load(f)
        

        return params

    
    
def main():
    args = arguments()

    params = read_config_params(args.config)

    model = MODELS[params["model"]]

    net = model()

    x = torch.randn(1, 3, 227, 227)

    print(net(x).shape)





if __name__ == "__main__":
    main()