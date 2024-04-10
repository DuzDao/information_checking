import transformers
import argparse
import yaml

from tasks.train import main as train_main
from tasks.predict import main as predict_main

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", choices=["train", "predict"], required=True)
    args = parser.parse_args()

    with open("ER_config.yaml") as conf_file:
        config = yaml.safe_load(conf_file)
    
    if args.tasks =="train":
        train_main(config)
    else:
        predict_main(config)  
    
