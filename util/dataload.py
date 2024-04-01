import yaml
import os
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import torch

CONFIG_PATH = 'config/config.yml'
CASE_PATH = 'config/case.yml'
GROUP_PATH = 'config/groups.yml'
PROP_MODEL_DIR = 'models/'
model_suffix = '_model.sav'
prop_models = None
root_dir = Path().parent.resolve()

idx_constraint_map = {
    # Mapping of property/constraint parameter in human readable format to index
    "NUM_GROUPS": 1,  # total number of groups
    "NUM_REPEAT_GROUPS": 2,  # total number of repeat groups
    "NUM_FUNC_GROUPS": 3,  # total number of functional groups
    "MOLECULAR_WEIGHT": 4,  # molecular weight
    "FLASH_POINT": 5,  # - flash point
    "MELTING_POINT":  6,  # normal melting point (Tm)
    "BOILING_POINT": 7,  # normal boiling point (Tb)
    "CRITICAL_TEMP":  8,  # critical temperature (Tc)
    "CRITICAL_PRESSURE":  9,  # critical pressure (Pc)
    "CRITICAL_VOLUME": 10,  # critical volume (Vc)
    "GIBBS":  11,  # standard Gibbs free energy, 298K
    "ENTHALPY_FORMATION": 12,  # standard Enthalpy of formation, 298K
    "ENTHALPY_VAP": 13,  # enthalpy of vaporization, 298K
    "ENTHALPY_FUS": 14,  # enthalpy of fusion, 298K
    "ENTHALPY_VAP_BP": 15,  # enthalpy of vaporization at Tb
    "AUTOIG_TEMP": 16,  # autoignition temp
    "PKA": 17,  # pKa
    "ENTHALPY_SOL": 18,  # heat of solution at P, hsol
    "LC50": 19,  # LC50
    "LOGP": 20,  # logP
    "LOGWS": 21,  # logWs
    "LD50": 22,  # ld50
    "OSHA_TWA": 23,  # osha-twa
    "HSP": 24,  # hild solubility param
    "BCF": 25,
    "VAPOR_PRESSURE": 26,  # vapor pressure
    "VISCOSITY": 27  # v iscosity
}


def open_yml(path):
    filepath = root_dir / Path(path)
    filepath = filepath.resolve()
    with filepath.open() as f:
        return yaml.safe_load(f)


def open_config():
    return open_yml(CONFIG_PATH)


def open_cases():
    return open_yml(CASE_PATH)


def open_groups():
    return open_yml(GROUP_PATH)


def open_dataset():
    data_dir = os.path.realpath(os.path.join(
        os.path.dirname(__file__), '..', 'data', 'valence.xlsx'))
    df = pd.read_excel(data_dir)
    df2 = df.copy()
    df.fillna(0, inplace=True)
    dataset = df.to_numpy()

    return df2, dataset


def load_prop_models(case, level=1):
    # refactor to load models ONCE when case suite data is loaded
    if (level == 1):
        filedir = os.path.join(root_dir, PROP_MODEL_DIR, 'first_order/')
    if (level == 2):
        filedir = os.path.join(root_dir, PROP_MODEL_DIR, 'second_order/')
    if (prop_models != None):
        return prop_models
    pls = {}
    for _, k in enumerate(case.constraints):
        # print(k)
        # no property model for MW determination
        if k[0] == 4:
            pls[4] = None
        if k[0] > 4:
            filename = filedir + \
                list(idx_constraint_map.keys())[int(k[0])-1] + model_suffix
            with open(filename, 'rb') as f:
                pl = pickle.load(f)
                pls[int(k[0])] = pl

    # print(pls.keys())
    return pls


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
