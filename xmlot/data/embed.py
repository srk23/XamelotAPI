# Transform the data into an ML-ready shape.

from xmlot.data.encode         import OneHotEncoder
from xmlot.data.dataframes     import get_sparse_columns
from xmlot.misc.lists import difference


############################
#     EMBEDDING STEPS      #
############################


def select_targets(df, targets_to_focus, descriptor):
    """
    Remove any row with any unknown target. Unused targets are removed.

    Args:
        - df               : the input DataFrame;
        - targets_to_focus : a set of targets we want to focus on;
        - descriptor       : a Descriptor.

    Returns:
        A DataFrame with all targets but the one mentionned dropped ;
        any row that contains missing value regarding the selected target is dropped as well.
    """
    targets_to_ignore = difference(
        [column for column in df.columns if descriptor.get_entry(column).tags == "target"],
        targets_to_focus
    )

    return df.drop(columns=targets_to_ignore) \
             .dropna(axis=0, subset=targets_to_focus, how='any')


def extract_dense_dataframe(df, threshold):
    """
    Remove the columns that are less dense than a given threshold,
    then, remove the rows that contain missing values.

    Args:
        - df        : the input DataFrame;
        - threshold : the density threshold.

    Returns:
        A dense DataFrame.
    """
    sparse_columns = get_sparse_columns(df, threshold)
    return df.drop(columns=sparse_columns) \
             .dropna()


def encode_categorical_data(
        df,
        descriptor,
        separator,
        exceptions,
        default_categories
):
    """
    Build a One Hot Encoder and encode categorical data with it.

    Args:
        - df                        : the input DataFrame;
        - descriptor                : the descriptor used for the initialization of the OHE;
        - encode_parameters_manager : contains the relevant parameters to the encoding operation.

    Returns:
        A one-hot encoded DataFrame, the One Hot Encoder.
    """
    ohe = OneHotEncoder(
        descriptor,
        separator,
        exceptions,
        default_categories
    )
    return ohe.encode(df), ohe

######################
#      VISITORS      #
######################
# Visitors allow to enrich functions (here 'embed_data') with optional behaviours.


class DefaultEmbedDataVisitor:
    def start(
        self,
        df,
        threshold,
        descriptor,
        separator,
        exceptions,
        default_categories,
        targets_to_focus
    ): pass

    def select_targets(self, df, target, encode_parameters_manager): pass

    def extract_dense_dataframe(self, df, threshold): pass

    def encode_categorical_data(
        self,
        df,
        descriptor,
        separator,
        exceptions,
        default_categories
    ):
        pass

class TalkativeEmbedDataVisitor(DefaultEmbedDataVisitor):
    def start(
        self,
        df,
        threshold,
        descriptor,
        separator,
        exceptions,
        default_categories,
        targets_to_focus
    ):
        string_output  = "Starting embedding...\n"
        string_output += "\tBefore embedding, the dataset has {0} rows and {1} columns.".format(*df.shape)

        print(string_output)

    def select_targets(self, df, targets_to_focus, descriptor):
        string_output  = "\nSelecting targets...\n"
        string_output += "\tWe focus on the following targets: `{0}`.\n".format(
            targets_to_focus
        )
        string_output += "\tThen, the dataset has {0} rows and {1} columns.".format(*df.shape)

        print(string_output)

    def encode_categorical_data(
        self,
        df,
        descriptor,
        separator,
        exceptions,
        default_categories
    ):
        string_output  = "\nFinalizing embedding...\n"
        string_output += "\tAt this point, the dataset has {0} rows and {1} columns.".format(*df.shape)

        print(string_output)


#################
#     EMBED     #
#################


def embed_data(
        df,
        threshold,
        descriptor,
        separator,
        exceptions,
        default_categories,
        targets_to_focus,
        visitor=DefaultEmbedDataVisitor()
):
    """
    Prepare the data to be injected into a machine learning model.
    Notably, it:
        - Remove unused targets;
        - Extract a dense DataFrame from the input one;
        - One Hot Encode categorical columns;
        - Convert types to `float32`.

    Args:
        - df                        : an input DataFrame;
        - threshold                 : the density threshold;
        - descriptor                : values that are generally used to mark unknown values;
        - encode_parameters_manager : an EncodeParametersManager for the one hot encoding step;
        - targets_to_focus          : targets
        - visitor                   : a visitor for additional/optional functionalities.

    Returns:
        - A machine learning ready to use version of the DataFrame provided as input.
        - The corresponding One Hot Encoder.
    """
    df = df.copy()
    visitor.start(
        df,
        threshold,
        descriptor,
        separator,
        exceptions,
        default_categories,
        targets_to_focus
    )

    df = select_targets(df, targets_to_focus, descriptor)
    visitor.select_targets(df, targets_to_focus, descriptor)

    # df = extract_dense_dataframe(df, threshold)
    # visitor.extract_dense_dataframe(df, threshold)

    df, ohe = encode_categorical_data(df, descriptor, separator, exceptions, default_categories)
    visitor.encode_categorical_data(df, descriptor, separator, exceptions, default_categories)

    return df, ohe
