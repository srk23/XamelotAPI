def intersection(l1, l2):
    return [e for e in l1 if e in l2]

def union(l1, l2):
    output = l1 + intersection(l2, l1)
    for e in l2:
        if e not in l1:
            output.append(e)
    return output

def difference(l1, l2):
    return [e for e in l1 if e not in l2]
