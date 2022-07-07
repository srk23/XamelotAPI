# Provide a suite of tools to describe and analyze data.

import matplotlib.pyplot          as     plt
import matplotlib.patches         as     phs
from   matplotlib.ticker          import MaxNLocator
import pandas                     as     pd
from   pprint                     import pprint

from   project.misc.dataframes    import density
from   project.misc.geometry      import get_distance_regarding_intersection, get_radius


class Entry:
    def __init__(
            self,
            column,
            description="",
            files="",
            column_type="",
            is_categorical=None,
            binary_keys=None,
            tags=""
    ):
        self.m_column         = column
        self.m_description    = description
        self.m_files          = files
        self.m_type           = column_type
        self.m_is_categorical = is_categorical
        self.m_binary_keys    = binary_keys
        if binary_keys:
            self.m_binary_vals = {v: k for (k, v) in binary_keys.items()}
        else:
            self.m_binary_vals = None
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
    def binary_keys(self):
        return self.m_binary_keys

    @property
    def binary_values(self):
        return self.m_binary_vals

    @property
    def is_binary(self):
        return bool(self.m_binary_keys)

    @property
    def tags(self):
        return self.m_tags

    def __repr__(self):
        categorical = "categorical" if self.is_categorical else "numerical"

        return """
{0}: 
    > {1}
    > This column is {2} (type: {3})
    > It belongs to files: {4}.
    > It has been tagged as: {5}.
""". format(
            self.column,
            self.description,
            categorical,
            self.type,
            self.files,
            self.tags
        )


class Descriptor:
    def __init__(self, entries):
        self.m_entries     = {entry.column: entry for entry in entries}

    def set_entry(self, entry):
        self.m_entries[entry.column] = entry

    def get_entry(self, column):
        return self.m_entries[column]

    def get_keys(self):
        return self.m_entries.keys()


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
        stats['count']   = pd.value_counts(df[column])
    else:
        stats['density'] = density(df, column)
        stats['min']     = df[column].min()
        stats['max']     = df[column].max()
        stats['mean']    = df[column].mean()
        stats['std']     = df[column].std()

    return stats


def plot_histogram(column, df, descriptor, bins=100):
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
    fig, ax = plt.subplots(figsize=(27, 9))
    s = df[column]

    # Categorical data
    if entry.is_categorical:
        # Display counts for each category
        c = pd.value_counts(s)
        x = c.index
        y = c.values

        # Yes/no answers are always displayed in the same order.
        if entry.is_binary:

            x = x.to_list()
            x.sort(reverse=False, key=lambda col: entry.binary_keys[col])
            y = [c[value] for value in x]

        # When the text values are too long to be displayed horizontally,
        # rotate them.
        if column in {'dcod'}:
            fig.autofmt_xdate(rotation=15)

        plt.bar(x, y)

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

        rect = phs.Rectangle(
            (left, bottom),
            width,
            top,
            alpha=0.1,
            facecolor="blue",
            label="mean ± std; std=" + str(s2), zorder=-5)
        plt.gca().add_patch(rect)

        # Display legend
        plt.legend()

    plt.show()


def print_summary(column, df, descriptor):
    """
    Provide a detailed inspection / description of the content of a column.
    """
    print(descriptor.get_entry(column))

    plot_histogram(column, df, descriptor)

    stats = get_stats(column, df, descriptor)
    if descriptor.get_entry(column).is_numerical:
        pprint(stats)
    else:
        print('density: %s' % stats['density'])
        pprint(pd.value_counts(df[column]))


def plot_densities(df):
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
    colors = ['orange' if y_ == 1 else 'C0' if y_ > .5 else 'red' for y_ in y]

    # Plot
    fig, ax = plt.subplots(figsize=(27, 9))
    ax.grid(visible=True, which='both', axis='y', zorder=0)
    ax.bar(x, y, color=colors, zorder=3)
    fig.autofmt_xdate(rotation=90)
    plt.show()

    # Print textual details
    print("Lowest density: %s (reached by %s).\n\n" % (y[-1], x[-1]) + "*" * 100 + "\n")
    print("Orange bars show densities equal to 1.")
    print("Red bars show columns that have more missing values than non-missing ones.")


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
