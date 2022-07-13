# Allow to save and load data/descriptors.

import pickle
import re

import pandas as pd

from project.config import Configurator
from project.data.describe import Entry, Descriptor
from project.misc.miscellaneous import string_autotype


def save_data(dataframes, config: Configurator, dump_name):
    """
    Save data into a serialized DataFrame.
    
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
    Load original data for .xlsx files into Pandas DataFrames.
    
    If specified, serialize and save these DataFrames locally.
    For organization purposes, serialized DataFrames are stored in a directory 
    called "Data" at the root of the project.
    
    Args:
        - config    : Configurator managing project's paths;
        - files     : a dictionary of shape {
                                'offering'  : list({offering_filenames}),
                                'transplant': list({transplant_filenames})
                      };
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
    Load data from serialized DataFrames.
    
    Args:
        - config    : Configurator managing project's paths;
        - dump_name : filename of the serialized DataFrames (no extension).
    
    Returns: a dictionary filled with DataFrames.
    """

    # Load data (from serialized dataframes)
    file = open(config.dump_dir + dump_name + ".pkl", 'rb')
    dataframes = pickle.load(file)
    file.close()

    # Focus on the most recent files
    return dataframes


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

        if not pd.isna(row['binary_keys']):
            binary_keys = re.split(':', row['binary_keys'])

            def retype(s):
                adjust_type, _ = string_autotype(s)
                return adjust_type(s)

            binary_keys = {retype(binary_keys[i]): i for i in range(2)}
        else:
            binary_keys = dict()

        new_entry = Entry(
            column=row['variable'],
            description=row['description'],
            files=row['files'],
            column_type=row['type'],
            is_categorical=row['is_categorical'],
            binary_keys=binary_keys,
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
    def _write_binary_keys_(binary_keys):
        if binary_keys:
            return '%s:%s' % tuple(binary_keys.keys())
        else:
            return ''

    csv_content = "variable,description,type,is_categorical,binary_keys,files,tags\n"

    for key in descriptor.get_keys():
        entry = descriptor.get_entry(key)
        csv_content += '"%s","%s","%s","%s","%s","%s","%s"\n' % (
            key,
            entry.description,
            entry.type,
            str(entry.is_categorical).upper(),
            _write_binary_keys_(entry.binary_keys),
            entry.files,
            entry.tags
        )

    f = open(config.description_dir + csv_name + ".csv", 'w')
    f.write(csv_content)
    f.close()