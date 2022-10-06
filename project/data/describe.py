# Introduce the idea of Descriptor to hold metadata related to the data itself.

class Entry:
    def __init__(
            self,
            column,
            description="",
            files="",
            column_type="",
            is_categorical=None,
            categorical_keys=None,
            tags=""
    ):
        self.m_column         = column
        self.m_description    = description
        self.m_files          = files
        self.m_type           = column_type
        self.m_is_categorical = is_categorical
        self.m_categorical_keys = categorical_keys
        if categorical_keys:
            self.m_categorical_vals = {v: k for (k, v) in categorical_keys.items()}
        else:
            self.m_categorical_vals = None
        self.m_tags           = tags

    @property
    def column(self):
        return self.m_column

    @property
    def description(self):
        return self.m_description

    @property
    def files(self):
        return self.m_files

    @property
    def type(self):
        return self.m_type

    @property
    def is_categorical(self):
        return self.m_is_categorical

    @property
    def is_numerical(self):
        return not self.m_is_categorical

    @property
    def categorical_keys(self):
        return self.m_categorical_keys

    @property
    def categorical_values(self):
        return self.m_categorical_vals

    @property
    def is_binary(self):
        if bool(self.categorical_keys):
            return len(self.categorical_keys) == 2
        return False

    @property
    def tags(self):
        return self.m_tags

    def __repr__(self):
        categorical = "categorical" if self.is_categorical else "numerical"

        output  = "{0}:\n".format(self.column)
        output += "\t> {0}\n".format(self.description)
        output += "\t> This column is {0} (type: {1}).\n".format(categorical, self.type)
        if self.is_binary:
            maxl = max(list(map(len, self.categorical_keys.keys())))
            for k, v in self.categorical_keys.items():
                output += "\t\t> {0}{1} : {2}\n".format(k, " " * (maxl - len(k)), v)
        output += "\t> It belongs to files: {0}.\n".format(self.files)
        output += "\t> It has been tagged as: {0}.\n".format(self.tags)

        return output


class Descriptor:
    def __init__(self, entries):
        self.m_entries     = {entry.column: entry for entry in entries}

    def set_entry(self, entry):
        self.m_entries[entry.column] = entry

    def get_entry(self, column):
        return self.m_entries[column]

    def get_keys(self):
        return self.m_entries.keys()


##################
#      MISC      #
##################

# Update Descriptor according to wrangling
def update_descriptor_after_wrangle(descriptor, files="new"):
    """
    Update a descriptor by adding the new columns introduced with `wrangle_data`.
    """
    for key in ['alt', 'ast', 'amylase', 'creatinine', 'degfr']:
        descriptor.set_entry(Entry(
            key + "_trend",
            description="Trend for %s." % key,
            files=files,
            column_type="object",
            is_categorical=True,
            categorical_keys="",
            tags="feature"
        ))

        descriptor.set_entry(Entry(
            key + "_min",
            description="Minimum value for %s." % key,
            files=files,
            column_type="float32",
            is_categorical=False,
            categorical_keys="",
            tags="feature"
        ))

        descriptor.set_entry(Entry(
            key + "_max",
            description="Maximum value for %s." % key,
            files=files,
            column_type="float32",
            is_categorical=False,
            categorical_keys="",
            tags="feature"
        ))

    descriptor.set_entry(Entry(
        "dial_type",
        description="Tells the most recent type of dialysis regarding transplantation.",
        files=files,
        column_type="object",
        is_categorical=True,
        categorical_keys="",
        tags="feature"
    ))

    descriptor.set_entry(Entry(
        "dial_days",
        description="Tells how long the patient have been on dialysis.",
        files=files,
        column_type="float32",
        is_categorical=False,
        categorical_keys="",
        tags="feature"
    ))
