# Allow to save and load various kind of data: datasets, parameters, descriptors, etc.

import json
import pickle
import re

import pandas as pd

from copy import deepcopy

from project.config                  import Configurator
from project.data.describe           import Entry, Descriptor
from project.data.parameters_manager import CleanParametersManager, EncodeParametersManager
from project.misc.miscellaneous      import identity, string_autotype


def save_data(dataframes, config: Configurator, dump_name):
    """
    Save data into a serialized DataFrame.
    Technically can save anything in the corresponding "dump" directory.

    Args:
        - dataframes: dataframes to save;
        - config    : Configurator managing project's paths;
        - dump_name : filename of the serialized DataFrame (no extension).
    
    Returns: None.
    """
    file = open(config.dump_dir + dump_name + ".pkl", 'wb')
    pickle.dump(dataframes, file)
    file.close()


def load_xlsx_data(config: Configurator, dump_name=""):
    """
    Load original data from .xlsx files into Pandas DataFrames.
    
    If specified, serialize and save these DataFrames locally.
    The place where serialized DataFrames are stored is specified by a Configurator.
    
    Args:
        - config    : Configurator managing project's paths;
        - dump_name : filename of the serialized DataFrames (no extension).
    
    Returns: similarly to files, return dictionary filled with the corresponding
    DataFrames.
    """

    # Load data (from original .xlsx files)
    dataframes = {
        'offering'  : list(map(lambda f: pd.read_excel(config.data_dir + f, sheet_name=1),
                               config.data_files['offering'])),
        'transplant': list(map(lambda f: pd.read_excel(config.data_dir + f, sheet_name=1),
                               config.data_files['transplant'])),
    }

    # Serialization
    if dump_name:
        save_data(dataframes, config, dump_name)
    else:
        print("Warning: Serialization disabled.")

    return dataframes


def load_data(config: Configurator, dump_name):
    """
    Load data from serialized DataFrame(s).
    Technically can load any serialized item in the corresponding "dump" directory.
    
    Args:
        - config    : Configurator managing project's paths;
        - dump_name : filename of the serialized DataFrames (no extension).
    
    Returns: a dictionary filled with DataFrames or a DataFrame.
    """
    file = open(config.dump_dir + dump_name + ".pkl", 'rb')
    data = pickle.load(file)
    file.close()
    return data


def load_descriptor(config: Configurator, csv_name):
    """
    Load descriptor from a .csv file.

    Args:
        - config   : Configurator managing project's paths;
        - csv_name : filename of .csv file.

    Returns: a Descriptor.
    """
    descriptor = Descriptor(dict())

    df = pd.read_csv(config.description_dir + csv_name + ".csv")

    for i in df.index:
        row = df.loc[i]

        if not pd.isna(row['categorical_keys']):
            categorical_keys = re.split(':', row['categorical_keys'])

            def retype(s):
                adjust_type, _ = string_autotype(s)
                return adjust_type(s)

            categorical_keys = {retype(categorical_keys[i]): i for i in range(len(categorical_keys))}
        else:
            categorical_keys = dict()

        new_entry = Entry(
            column=row['variable'],
            description=row['description'],
            files=row['files'],
            column_type=row['type'],
            is_categorical=row['is_categorical'],
            categorical_keys=categorical_keys,
            tags=row['tags']
        )
        descriptor.set_entry(new_entry)

    return descriptor


def save_descriptor(descriptor, config: Configurator, csv_name):
    """
    Save a Descriptor into a .csv file.

    Args:
        - descriptor: descriptor to save;
        - config    : Configurator managing project's paths;
        - csv_name  : filename of the serialized DataFrame (no extension).

    Returns: None.
    """
    def _write_categorical_keys_(categorical_keys):
        if categorical_keys:
            return ("%s:" * len(categorical_keys))[:-1] % tuple(categorical_keys.keys())
        else:
            return ""

    # Build .csv content
    csv_content = "variable,description,type,is_categorical,categorical_keys,files,tags\n"

    for key in descriptor.get_keys():
        entry = descriptor.get_entry(key)
        csv_content += '"%s","%s","%s","%s","%s","%s","%s"\n' % (
            key,
            entry.description,
            entry.type,
            str(entry.is_categorical).upper(),
            _write_categorical_keys_(entry.categorical_keys),
            entry.files,
            entry.tags
        )

    # Write content into a file
    f = open(config.description_dir + csv_name + ".csv", 'w')
    f.write(csv_content)
    f.close()

def load_clean_parameters_manager(config: Configurator, json_name):
    """
    Load parameters related to the cleaning step from a .json file.

    Args:
        - config    : Configurator managing project's paths;
        - json_name : name of the .json file (no extension).

    Returns:
        A CleanParametersManager
    """
    # Load .json
    with open(config.parameters_dir + json_name + ".json") as parameters_json:
        parameters = json.load(parameters_json)

    # Build CleanParametersManager
    cpm = CleanParametersManager(
        heterogeneous_columns = parameters["HETEROGENEOUS_COLUMNS"],
        generic_unknowns      = parameters["GENERIC_UNKNOWNS"],
        specific_unknowns     = parameters["SPECIFIC_UNKNOWNS"],
        limits                = parameters["LIMITS"],
        bmi_limits            = parameters["BMI_LIMITS"],
        references            = parameters["REFERENCES"],
        categorical_keys      = parameters["CATEGORICAL_KEYS"],
        replacement_pairs     = parameters["REPLACEMENT_PAIRS"],
        columns_to_categorise = parameters["COLUMNS_TO_CATEGORISE"],
        irrelevant_categories = parameters["IRRELEVANT_CATEGORIES"],
        irrelevant_columns    = parameters["IRRELEVANT_COLUMNS"],
        columns_with_unknowns = parameters["COLUMNS_WITH_UNKNOWNS"],
        unknown               = parameters["UNKNOWN"]
    )

    # Since JSON does not handle int as keys, we need to do it "by hand".
    typed_references = deepcopy(cpm.references)
    for i, ref_group in enumerate(cpm.references):
        _, reference = ref_group

        for key in cpm.references[i][1].keys():
            if  re.fullmatch("[0-9]+", key) and cpm.references[i][0][0] != 'mgrade':
                adjust_type = int
            else:
                adjust_type = identity
            typed_references[i][1][adjust_type(key)] = typed_references[i][1].pop(key)
    cpm.references = typed_references

    return cpm

def load_encode_parameters_manager(config: Configurator, json_name):
    """
    Load parameters related to the encoding step from a .json file.

    Args:
        - config    : Configurator managing project's paths;
        - json_name : name of the .json file (no extension).

    Returns:
        A EncodeParametersManager
    """
    with open(config.parameters_dir + json_name + ".json") as parameters_json:
        parameters = json.load(parameters_json)

    return EncodeParametersManager(
        separator          = parameters["SEPARATOR"],
        exceptions         = parameters["EXCEPTIONS"],
        default_categories = parameters["DEFAULT_CATEGORIES"],
    )
