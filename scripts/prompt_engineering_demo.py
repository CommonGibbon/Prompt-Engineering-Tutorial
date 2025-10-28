import os
import pandas as pd
from pathlib import Path
import mlflow
from omegaconf import DictConfig
import hydra

from prompt_engineering.utils import CatIdentifier, compare_accuracy

# hydra uses .yaml config files to track what settings you want to use for a particular run. To change our run, all we need to do is edit /conf/config.yaml accordingly
# and the settings will be loaded and used throughout our experiment script.
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    cat_identifier = CatIdentifier(cfg.model) # initialize and instance of our cat identifier, defined in /src/prompt_engineering/utils.py

    label_df = pd.read_csv(cfg.label_path) # manualy generated labels are stored locally
    
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri) # this determines where our ml run data is stored. We're using a local folder here.
    mlflow.set_experiment(cfg.mlflow.experiment_name) # set an experiment name. Our goal is prompt optimization for cat identification.
    
    with mlflow.start_run(run_name = cfg.mlflow.run_name): # mlflow will create a new run folder to track this run

        mlflow.log_param("prompt_version", cfg.prompts.version) # track which version of the prompt we used
        mlflow.log_param("prompt_description", cfg.prompts.system_prompt) # track the description we configured for this prompt

        # make the predicitons. Note (if you followed along in the notebook tutorial) that this uses identify_comp and provides a sample image.
        preds = {image_id: cat_identifier.identify_comp(cfg.sample_image_path,f"{cfg.image_path}/{image_id}.jpg", cfg.prompts.system_prompt)["cat"] for image_id in label_df.image_id}

        acc = compare_accuracy(preds, label_df) # evaluate the predictions against the labels

        mlflow.log_metric("Accuracy", acc) # track the accuracy of our predictions

if __name__ == "__main__":
    main()