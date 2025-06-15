import os
import yaml
from box import ConfigBox
from typing import Any
from box.exceptions import BoxValueError
from pathlib import Path
from ensure import ensure_annotations
from src.logging.logger import logger
import pickle

@ensure_annotations
def read_yaml(yaml_path:Path)->ConfigBox:
    try:
        with open(yaml_path) as f:
            content=yaml.safe_load(f)
            logger.info(f" yaml file {yaml_path} loaded successfully!")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"yaml file {os.path.split(yaml_path)[1]} is empty!")
@ensure_annotations   
def write_yaml_file(path_to_yaml: Path, data:dict)->None:
    try:
        Path(path_to_yaml).parent.mkdir(parents= True,exist_ok=True)
        with open(path_to_yaml,"w") as yaml_file:
            yaml.dump(data,yaml_file,default_flow_style=False)
            logger.info(f"YAML file written to {path_to_yaml}")
    except Exception as e:
        logger.error(f"Failed to write YAML file to {path_to_yaml}: {e}")
        raise e
@ensure_annotations
def save_bin(file_path:str,data:Any)->None:
    try:
        os.makedirs(Path(file_path).parent,
                    exist_ok=True)
        with open(file_path,"wb") as f:
            pickle.dump(data,f)
        logger.info("Object was saved successfully")
    except Exception as e:
        raise e
    
@ensure_annotations
def load_bin(file_path:str,data:Any)->None:
    try:
        os.makedirs(Path(file_path).parent,
                    exist_ok=True)
        with open(file_path,"wb") as f:
            pickle.load(data,f)
        logger.info("Object was loaded successfully")
    except Exception as e:
        raise e

    

