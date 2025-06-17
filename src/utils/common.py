import os
import yaml
from box import ConfigBox
from typing import Any
import numpy as np
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
def save_bin(file_path:str,data:Any)->None:
    try:
        os.makedirs(Path(file_path).parent,
                    exist_ok=True)
        with open(file_path,"wb") as f:
            pickle.dump(data,f)
        logger.info("Object was saved successfully")
    except Exception as e:
        raise e
def load_bin(file_path:str)->Any:
    try:
        with open(file_path,"rb") as f:
            object =pickle.load(f)
            logger.info("Object was loaded successfully")
        return object
    except Exception as e:
        raise e

def save_array(file_path:str,data:np.ndarray)->None:
            try:
                if not file_path.endswith(".npy"):
                    raise ValueError("File path must end with '.npy' for saving NumPy arrays.")

                os.makedirs(Path(file_path).parent, exist_ok=True)
                np.save(file_path, data)
                logger.info(f" NumPy array saved successfully to {file_path}")

            except Exception as e:
                logger.error(f"Failed to save NumPy array to {file_path}: {e}")
                raise e

def load_array(file_path:str)->np.ndarray:
        try:
            if not file_path.endswith(".npy"):
                raise ValueError("File path must end with '.npy' for loading NumPy arrays.")

            array = np.load(file_path)
            logger.info(f" NumPy array loaded successfully from {file_path}")
            return array

        except Exception as e:
            logger.error(f" Failed to load NumPy array from {file_path}: {e}")
            raise e