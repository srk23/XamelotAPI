# Transform the data into a more ML-friendly shape.

from project.data.encode                import OneHotEncoder
from project.misc.dataframes            import get_sparse_columns
from project.data.survival_data_manager import SurvivalDataManager

######################
#      VISITORS      #
######################

class DefaultEmbedDataVisitor:
    def start(
        self,
        df,
        target,
        threshold,
        descriptor,
        separator,
    ): pass

    def select_targets(self, df, target, descriptor): pass

    def extract_dense_dataframe(self, df, threshold): pass

    def encode_categorical_data(self, df, descriptor, separator): pass

    def convert_to_float32(self, df): pass

class TalkativeEmbedDataVisitor(DefaultEmbedDataVisitor):
    def start(
        self,
        df,
        target,
        threshold,
        descriptor,
        separator,
    ):
        string_output  = "Starting embedding...\n"
        string_output += "\tBefore embedding, the dataset has {0} rows and {1} columns.".format(*df.shape)

        print(string_output)

    def select_targets(self, df, target, descriptor):
        string_output  = "\nSelecting targets...\n"
        string_output += "\tWe focus on the following targets: `{0}`, `{1}`.\n".format(*target)
        string_output += "\tThen, the dataset has {0} rows and {1} columns.".format(*df.shape)

        print(string_output)

    def convert_to_float32(self, df):
        string_output  = "\nFinalizing embedding...\n"
        string_output += "\tAt this point, the dataset has {0} rows and {1} columns.".format(*df.shape)

        print(string_output)


###################
#      EMBED      #
###################


def select_targets(df, target, descriptor):
    """
    Select targets to predict.
    Note: we are not putting ourselves in a competing survival analysis context.

    Args:
        - df         : the input DataFrame
        - target     : a pair of event and duration to focus on
        - descriptor : a Descriptor

    Returns:
        A DataFrame with all targets but the one mentionned dropped ;
        any row missing a value regarding the selected target is dropped as well.
    """
    target_event, target_duration = target

    targets_to_focus  = {target_event, target_duration}
    targets_to_ignore = {column for column in df if descriptor.get_entry(column).tags == "target"} - targets_to_focus

    return df.drop(columns=targets_to_ignore) \
             .dropna(subset=targets_to_focus)


def extract_dense_dataframe(df, threshold):
    """
    Remove the columns that are less dense than a given threshold,
    then remove the rows that contain missing values.

    Args:
        - df        : the input DataFrame
        - threshold : the density threshold

    Returns:
        A dense DataFrame
    """
    sparse_columns = get_sparse_columns(df, threshold)
    return df.drop(columns=sparse_columns) \
             .dropna()


def encode_categorical_data(df, descriptor, separator):
    """
    Build a One Hot Encoder and encode categorical data with it.

    Args:
        - df         : the input DataFrame
        - descriptor : the descriptor used for the initialization of the OHE
        - separator  : the separator used for the initialization of the OHE

    Returns:
        An "encoded" DataFrame, the One Hot Encoder
    """
    ohe = OneHotEncoder(descriptor, separator=separator)
    return ohe.encode(df), ohe


def convert_to_float32(df):
    """
    Convert `dtypes` to 'float32' (because of Pytorch requirements).
    """
    return df.astype('float32')

def embed_data(
        df,
        target,
        threshold,
        descriptor,
        separator='#',
        visitor=DefaultEmbedDataVisitor()
):
    """
    Prepare the data to be injected into a machine learning model.
    Notably, it:

    - Remove unused targets;
    - Extract a dense DataFrame from the input one;
    - One Hot Encode categorical columns;
    - Convert types to `float32`;
    - Return a SurvivalDataManager

    Args:
        - df         : an input DataFrame
        - target     : a pair of event and duration to focus on
        - threshold  : the density threshold
        - descriptor : values that are generally used to mark unknown values
        - separator  : values that are used to mark unknown values, per column
        - standardization_function    : the function that needs to be applied to the numerical data
        - standardize_target_duration : tells whether we should modify or not the targetted survival time

    Returns:
        - A machine learning ready to use version of the DataFrame provided as input.
        - The build One Hot Encoder.
    """
    df = df.copy()
    visitor.start(
        df,
        target,
        threshold,
        descriptor,
        separator,
    )

    df = select_targets(df, target, descriptor)
    visitor.select_targets(df, target, descriptor)

    df = extract_dense_dataframe(df, threshold)
    visitor.extract_dense_dataframe(df, threshold)

    df, ohe = encode_categorical_data(df, descriptor, separator)
    visitor.encode_categorical_data(df, descriptor, separator)

    df = convert_to_float32(df)
    visitor.convert_to_float32(df)

    return SurvivalDataManager(df, *target, ohe=ohe)
