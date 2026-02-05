import numpy as np

from pkg.newton.newton import newton_system
from pkg.newton.utils import plot_errors
from pkg.newton.test_functions import *

if __name__ == "__main__":

    tests = [q1c_f1, q1c_f2, q1c_f3]
    for i, testfun in enumerate(tests, 1):
        print(f"\nTEST FUNCTION {i}")
        F, J, x0, x_true = testfun()
        sol, info = newton_system(F, x0, J, x_true)
        print("solution:", sol)
        plot_errors(info["error_history"])


