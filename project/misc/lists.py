# Basic set theoretic operations


def intersection(l1, l2):
    return [e for e in l1 if e in l2]

def difference(l1, l2):
    return [e for e in l1 if e not in l2]

def union(l1, l2):
    return l1 + difference(l2, l1)
