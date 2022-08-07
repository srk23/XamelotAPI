class CleanParametersManager:
    def __init__(
            self,
            descriptor=None,
            heterogeneous_columns=None,
            generic_unknowns=None,
            specific_unknowns=None,
            limits=None,
            bmi_limits=None,
            references=None,
            binary_keys=None,
            irrelevant_categories=None,
            irrelevant_columns=None,
            columns_with_unknowns=None,
            unknown=None
    ):
        self.m_descriptor            = descriptor
        self.m_heterogeneous_columns = heterogeneous_columns
        self.m_generic_unknowns      = generic_unknowns
        self.m_specific_unknowns     = specific_unknowns
        self.m_limits                = limits
        self.m_bmi_limits            = bmi_limits
        self.m_references            = references
        self.m_binary_keys           = binary_keys
        self.m_irrelevant_categories = irrelevant_categories
        self.m_irrelevant_columns    = irrelevant_columns
        self.m_columns_with_unknowns = columns_with_unknowns
        self.m_unknown               = unknown

    @property
    def descriptor(self):
        return self.m_descriptor

    @descriptor.setter
    def descriptor(self, descriptor):
        self.m_descriptor = descriptor

    @property
    def heterogeneous_columns(self):
        return self.m_heterogeneous_columns

    @property
    def generic_unknowns(self):
        return self.m_generic_unknowns

    @property
    def specific_unknowns(self):
        return self.m_specific_unknowns

    @property
    def limits(self):
        return self.m_limits

    @property
    def bmi_limits(self):
        return self.m_bmi_limits

    @property
    def references(self):
        return self.m_references

    @references.setter
    def references(self, references):
        self.m_references = references

    @property
    def binary_keys(self):
        return self.m_binary_keys

    @binary_keys.setter
    def binary_keys(self, binary_keys):
        self.m_binary_keys = binary_keys

    @property
    def irrelevant_categories(self):
        return self.m_irrelevant_categories

    @property
    def irrelevant_columns(self):
        return self.m_irrelevant_columns

    @property
    def columns_with_unknowns(self):
        return self.m_columns_with_unknowns

    @property
    def unknown(self):
        return self.m_unknown
