
def identity(x, ref_value, is_encode):
    return x


def normalize(x, ref_value, is_encode):
    if is_encode:
        return x / ref_value - 1
    else:
        return (x + 1) * ref_value

