import numpy as np

from pkg.newton.newton import newton_system
from pkg.newton.utils import plot_errors, plot_newton_convergence, plot_residual_heatmap
from pkg.newton.test_functions import *

if __name__ == "__main__":

	# A. convergence tests for chosen functions.
    # tests = [q1c_f1, q1c_f2, q1c_f3]
    # labels = [r'$S_1$', r'$S_2$', r'$S_3$']
    # for i, (testfun, label) in enumerate(zip(tests, labels), 1):
    #     print(f"\nTEST FUNCTION {i}")
    #     F, J, x0, x_true = testfun()
    #     sol, info = newton_system(F, x0, J, x_true)
    #     print("solution:", sol)
    #     plot_errors(info["error_history"], title_suffix=label)
        
	
	# B. visual verification.
	# function N4.
	F, J, x0, x_true = q1b_visual_check_f1()
	plot_residual_heatmap(F, J, x0, x_true, title="Residual Heatmap")
	plot_newton_convergence(F, J, x0, x_true, title="Convergence History")
	
	# function N2.
	F, J, x0, x_true = q1b_visual_check_f2()
	plot_residual_heatmap(F, J, x0, x_true, title="Residual Heatmap", grid_bounds=(0, 3))
	plot_newton_convergence(F, J, x0, x_true, title="Convergence History")
    


