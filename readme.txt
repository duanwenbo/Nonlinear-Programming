-- Author: Wenbo Duan
-- Email: pv19120@bristol.ac.uk

################################################
1. Environment used:
- Python 3.8
- sympy 	 (install command: pip -install sympy)



################################################
2. How to run the program ?
- Type the objective function and initial position in optimizer.py under the entrance function 
- Run optimizer.py



################## important ###################
3. How to input quesion?

The following codes were captured from optimizater.py
-------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ######################## input the NLP question there ########################
    opt = Optimizer(
        objective_function="(x_1 - 1)**2 + (x_2 -2)**2 + (x_3 - 3)**2",
        initial_position="x_1=0.5, x_2=0.5, x_3 = 0.5",
        type="minimisation")
    ##############################################################################
    opt.conjugate_gradient()
-----------------------------------------------------------------------------------------------------------------------------------

- type the objective function behind "objective_function=" and within quotation marks.
- type the initial position behind "initial_position="  and within quotation marks.
- type the type of the optimisation behind "type="  and within quotation marks. The available choices are : "minimisation" or "maximisation"
################################################
4. File Interpretation

- optimizer.py:
main file to implement conjugate gradient

- log.txt: 
It records the intermediate parameters during optimization, including:
current position, current direction vector,  direction updating parameter beta ,  step length parameter alpha


- solution.txt: 
This log file records the optimal converge point found

For more details please refer to the original files




---Feel free to ask me for more info if anything happened during implementation---





