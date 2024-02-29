

def check_disjoint(lists):
    sets = [set(l) for l in lists]
    for i in range(len(sets)-1):
        for j in range(i+1, len(sets)):
            assert len(sets[i].intersection(sets[j])) == 0, \
                f'ERROR: Found an overlap between the {i+1}-th and {j+1}-th sets.'
    total_size = 0
    union = set()
    for s in sets:
        total_size += len(s)
        union.update(s)
    assert len(union) == total_size, f'ERROR: Thee sets are not disjoint.'