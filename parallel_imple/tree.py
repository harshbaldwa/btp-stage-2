def bits_interleaving(b1, b2):
    """
    Returns the bit formed after interleaving b1 and b2

    Args:
        b1 ([array]): [binary representation of a number, say x1]
        b2 ([array]): [binary representation of a number, x2]

    Returns:
        [array]: [the result of interleaving x1 and x2]
    """
    n = len(b1)
    inter = [None]*2*n
    inter[::2] = b1
    inter[1::2] = b2
    
    return inter


def decimal_to_binary(n, b_len):
    """
    Converts decimal to binary

    Args:
        n ([int]): [number to be converted]
        b_len ([int]): [length of binary representation]

    Returns:
        [array]: [binary representation]
    """
    binary = [None]*b_len
    for i in range(b_len):
        binary[i] = n%2
        n = n//2
    
    return binary[::-1]


def binary_to_decimal(b):
    """
    Converts binary to decimal

    Args:
        b ([array]): [binary representation]

    Returns:
        [int]: [decimal number]
    """
    num = 0
    n = len(b)
    for i in range(0, n):
        num += b[i]*2**(n-i-1)
    
    return num



def get_cell_id(x, y, level, x_min=0, y_min=0, length=1): 
    """
    Returns the ID of corresponding cell
    """
    b_len = 2**(level-1)
    nx = int(b_len*2*(x-x_min)/length)
    ny = int(b_len*2*(y-y_min)/length)
    bx = decimal_to_binary(nx, b_len)
    by = decimal_to_binary(ny, b_len)
    inter = bits_interleaving(bx, by)
    id = binary_to_decimal(inter)
    return id


def get_bin_count(x, y, level, bin_count, bin_offset):
    for i in range(len(bin_offset)):
        id = get_cell_id(x[i], y[i], level)
        bin_offset[i] = bin_count[id]
        bin_count[id] += 1
    

def cum_bin_count(bin_count, start):
    for i in range(1, len(bin_count)):
        start[i] = start[i-1] + bin_count[i-1]


def start_indices(x, y, level, bin_offset, start, indices):
    for i in range(len(bin_offset)):
        id = get_cell_id(x[i], y[i], level)
        indices[start[id] + bin_offset[i]] = i


if __name__ == "__main__":
    x = [0.25, 0.75, 0.75, 0.25, 0.2]
    y = [0.75, 0.25, 0.75, 0.25, 0.2]

    level = 1

    bin_count = [0]*4**level
    start = [0]*4**level
    bin_offset = [0]*len(x)
    indices = [None]*len(x)

    get_bin_count(x, y, level, bin_count, bin_offset)
    print(bin_count)
    # cum_bin_count(bin_count, start)
    # start_indices(x, y, level, bin_offset, start, indices)
    
    # print(bin_count)
    # print(start)
    # # print(bin_offset)
    # print(indices)

