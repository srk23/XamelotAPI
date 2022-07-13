# Functions to standardize numerical data
def standardize(s):
    return (s - s.mean()) / s.std()
