import math
from pkg.lagrange import lagrange


# ---- Function to generate equally spaced nodes ----
def generate_nodes(a, b, n):
    return [a + i*(b-a)/(n-1) for i in range(n)]