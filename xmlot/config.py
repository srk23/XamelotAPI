# This file allows paths management.

import json
import os


class Configurator:
    def __init__(self,
                 data_dir      = None,
                 main_dir      = None,
                 dump_dir      = None,
                 desc_dir      = None,
                 params_dir    = None,
                 data_files    = None
                 ):
        """
        Allow to provide paths to data files and main directory.
        """
        self.m_data_dir   = data_dir
        self.m_main_dir   = main_dir
        self.m_dump_dir   = dump_dir
        self.m_desc_dir   = desc_dir
        self.m_params_dir = params_dir
        self.m_data_files = data_files

    @property
    def data_dir(self):
        return self.m_data_dir

    @property
    def main_dir(self):
        return self.m_main_dir

    @property
    def dump_dir(self):
        return self.m_dump_dir

    @property
    def description_dir(self):
        return self.m_desc_dir

    @property
    def parameters_dir(self):
        return self.m_params_dir

    @property
    def data_files(self):
        return self.m_data_files


def easy_config(talkative=False):
    """
    Build a Configurator storing only
    """
    main_dir = os.getcwd() + "/"
    config = Configurator(
        main_dir=main_dir,
        dump_dir=main_dir   + "Data/dump/",
        desc_dir=main_dir   + "Data/desc/",
        params_dir=main_dir + "Data/params"
    )
    if talkative:
        print(
            """
            Easy configuration:
            MAIN_DIR   : %s
            DUMP_DIR   : %s
            DESC_DIR   : %s
            PARAMS_DIR : %s
            DATA_DIR   : empty
            DATA_FILES : empty
            """ % (config.main_dir, config.dump_dir, config.description_dir, config.parameters_dir))
    return config


def json_config(json_filename):
    """
        Build a Configurator from a .json file.
        The corresponding .json file should have a structure similar to that example:
        ```{
            "MAIN_DIR"  : "/home/user/documents/projects/xmlot/",
            "DUMP_DIR"  : "/home/user/documents/projects/xmlot/data/dump/",
            "DESC_DIR"  : "/home/user/documents/projects/xmlot/data/description/",
            "PARAMS_DIR": "/home/user/documents/projects/xmlot/data/",
            "DATA_DIR"  : "/mnt/netshare/projects/xmlot/data/",
            "DATA_FILES": {
                "offering": [
                    "offering_file_1.xlsx",
                    "offering_file_2.xlsx"
                ],
                "transplant": [
                    "transplant_file_1.xlsx"
                ]
            }
        }```
    """
    with open(json_filename) as config_json:
        config = json.load(config_json)

    return Configurator(
        data_dir      = config["DATA_DIR"],
        main_dir      = config["MAIN_DIR"],
        dump_dir      = config["DUMP_DIR"],
        desc_dir      = config["DESC_DIR"],
        params_dir    = config["PARAMS_DIR"],
        data_files    = config["DATA_FILES"]
    )
