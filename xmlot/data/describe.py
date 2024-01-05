# Provides a suite of tools to describe data based on meta-datas, visualisation, etc.

import matplotlib.pyplot          as     plt
import matplotlib.patches         as     phs
from   matplotlib.ticker          import MaxNLocator
import pandas                     as     pd

from   xmlot.data.dataframes    import density, get_sparse_columns
from   xmlot.misc.geometry      import get_distance_regarding_intersection, get_radius


#####################
#    DESCRIPTOR     #
#####################
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
        if categorical_keys is not None:
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


#####################
#  TEXTUAL DETAILS  #
#####################


def print_whole_dataframe(df):
    with pd.option_context("display.max_rows", 1000):
        print(df)


def get_stats(column, df, descriptor):
    """
    Give some stats about a given column (regarding a specific dataframe).
    If the data is categorical, it returns:
    - density   : the proportion of defined data regarding the whole data.
    - count     : the count for each category.
    If the data is numerical, it returns:
    - density   : the proportion of defined data regarding the whole data.
    - min, max  : minimum and maximum values contained in that column.
    - mean, std : mean and standard deviation of the values contained in that column.
    """
    stats = dict()

    # Check if it is a column from the specified dataframe
    if column not in df.columns:
        raise KeyError

    # Check if the column name is known
    # Get information about it
    entry = descriptor.get_entry(column)

    # Get statistics
    if entry.is_categorical:
        stats['density'] = density(df, column)
        stats['count']   = df[column].value_counts()
    else:
        stats['density'] = density(df, column)
        stats['min']     = df[column].min()
        stats['max']     = df[column].max()
        stats['mean']    = df[column].mean()
        stats['std']     = df[column].std()

    return stats


def print_summary(df, descriptor):
    """
    Provide a detailed inspection / description of the content of a column.
    """
    ctr = 1
    for column in df.columns:
        entry = descriptor.get_entry(column)
        stats = get_stats(column, df, descriptor)
        print(entry)
        if entry.is_numerical:
            for k, v in stats.items():
                print("{0}{1} : {2}".format(k, ' ' * (7 - len(k)), v))
        else:
            print("Density: %s" % stats["density"])
        plot_histogram(column, df, descriptor)
        if ctr < len(df.columns):
            ctr += 1
            print('\n' + '#' * 50 + '\n')


#####################
#   VISUALISATION   #
#####################


def plot_histogram(column, df, descriptor, bins=100,figsize=(27, 9), rotate=False):
    """
    Plot the histogram/counted categories for a given column in a specific dataset.
    """

    # Check if the column name is known
    # Get information about it
    entry = descriptor.get_entry(column)

    # Check if it is a column from the specified dataframe
    if column not in df.columns:
        raise KeyError

    # Build figure according to column's type
    fig, ax = plt.subplots(figsize=figsize)
    s = df[column]

    # Categorical data
    if entry.is_categorical:
        # Display counts for each category
        c = s.value_counts()
        x = c.index
        y = c.values

        # Yes/no answers are always displayed in the same order.
        if entry.is_binary:

            x = x.to_list()
            x.sort(reverse=False, key=lambda col: entry.categorical_keys[col])
            y = [c[value] for value in x]

        # When the text values are too long to be displayed horizontally,
        # rotate them.
        if column in {'dcod', 'prd'} or rotate:
            fig.autofmt_xdate(rotation=15)
        plt.xticks(fontsize='xx-large')
        plt.bar(x, y)

        print(c)

    # Numerical data:
    else:
        # Force matplotlib to display integers on the x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Display hist
        s.hist(bins=bins, ax=ax, color='orange')

        # Display mean
        mu = s.mean()
        _, y_max = ax.get_ylim()
        ax.plot([mu, mu], [0, y_max], 'b--', label='mean=' + str(mu))

        # Display std
        s2 = s.std()

        left = mu - s2
        width = 2 * s2
        bottom = 0
        top = .97 * y_max

        # Display minimum and maximum
        mini = s.min()
        ax.plot([mini, mini], [0, y_max], 'r')

        maxi = s.max()
        ax.plot([maxi, maxi], [0, y_max], 'r', label='min={0}, max={1}'.format(mini, maxi))

        rect = phs.Rectangle(
            (left, bottom),
            width,
            top,
            alpha=0.1,
            facecolor="blue",
            label="mean ± std; std=" + str(s2), zorder=-5)
        plt.gca().add_patch(rect)

        # Display legend
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[0], handles[2], handles[1]]
        labels  = [labels[0] , labels[2] , labels[1] ]

        ax.legend(handles, labels, loc="upper right")

    plt.show()

def plot_densities(df, threshold=.5):
    """
    Display the bar plot of the densities of each column for a given DataFrame.
    """
    # Get columns and respective densities
    xy = (df.count() / len(df))

    # Sort values in decreasing order
    xy = xy.sort_values(ascending=False)

    # get lists to feed to Matplotlib
    x = xy.index.to_list()
    y = list(xy.values)

    # Define color scheme
    colors = ['orange' if y_ == 1 else 'C0' if y_ > threshold else 'red' for y_ in y]

    # Plot
    fig, ax = plt.subplots(figsize=(27, 9))
    ax.grid(visible=True, which='both', axis='y', zorder=0)
    ax.set_yticks([.1 * i for i in range(11)])
    ax.bar(x, y, color=colors, zorder=3)
    fig.autofmt_xdate(rotation=90)
    plt.show()

    # Print textual details
    print("Lowest density: %s (reached by %s).\n\n" % (y[-1], x[-1]) + "*" * 100 + "\n")
    print("Orange bars show densities equal to 1.")
    print("Red bars show columns that have more missing values than non-missing ones.")


def plot_thresholds(df, steps, row_yticks=(0, 40000, 5000), col_yticks=(0, 90, 10)):
    # Compute
    thresholds = [i / steps for i in range(steps + 1)]
    maxnrow, maxncol = df.shape
    ncols = []
    nrows = []

    for threshold in thresholds:
        # Note: it is basically 'extract_dense_dataframe' from 'xmlot.data.embed'.
        # However, using it triggers some circular import.
        df_ = df.copy().drop(columns=get_sparse_columns(df, threshold)).dropna()
        nrows.append(df_.shape[0])
        ncols.append(df_.shape[1])

    # Plot
    fig, ax_row = plt.subplots(figsize=(20, 10))
    ax_col = ax_row.twinx()

    lns_row = ax_row.plot(thresholds, nrows, color="orange", label="rows")
    lns_col = ax_col.plot(thresholds, ncols, label="columns")

    ax_row.grid(visible=True, which='both', zorder=0, color="orange" , alpha=.7)
    ax_col.grid(visible=True, which='both', zorder=0, color="#1f77b4", alpha=.5)

    ax_row.set_yticks(range(*row_yticks))
    ax_col.set_yticks(range(*col_yticks))
    plt.xticks([.05 * i for i in range(21)])

    ax_row.set_xlabel("Threshold")
    ax_row.set_ylabel("Number of rows (max: %s)"    % maxnrow)
    ax_col.set_ylabel("Number of columns (max: %s)" % maxncol)

    lns = lns_row + lns_col
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='lower left')
    plt.show()


def plot_intersection(df1, df2, label1='Transplant', label2='Offering'):
    """
    Provide a visual description of the number of columns that are common to two DataFrames.
    """
    # Get the list of columns for each DataFrame
    df1_col = list(df1.columns.values)
    df2_col = list(df2.columns.values)

    # Compute areas
    a1 = len(df1_col)
    a2 = len(df2_col)
    a3 = len(set(df1_col) & set(df2_col))

    # Compute distances
    r1 = get_radius(a1)
    r2 = get_radius(a2)
    d = get_distance_regarding_intersection(a1, a2, a3)

    # Plot
    circle1 = plt.Circle((0, 0), r1, color='orange', alpha=1 , clip_on=False, label=label1)
    circle2 = plt.Circle((d, 0), r2, color='C0'    , alpha=.7, clip_on=False, label=label2)

    fig, ax = plt.subplots(figsize=(27, 6))

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    plt.legend()
    plt.axis('equal')
    plt.axis('off')
    plt.show()

    # Print textual details
    print(
        """
        Total number of columns in 'Transplant' : %s
        Total number of columns in 'Offering'   : %s
        Number of mutual columns                : %s
        """ % (a1, a2, a3)
    )
