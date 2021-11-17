import sympy as sp  # tools for matrix, calculus operation

####################################################
"""set the objective function there"""
# 1. declare variables
x_1, x_2 = sp.symbols('x_1 x_2') 
# 2. initialize obj func
f = -x_1**2 + 4*x_1 + 2*x_1*x_2 - 2*x_2**2 
# 3. declare init position
starting_position = [(x_1,0), (x_2,0)]  
#####################################################
alpha = sp.symbols('alpha')  # step lenghth 



def _update_position(position_vector, current_position, step_length = alpha):
    assert len(starting_position) == sp.shape(position_vector)[1], "check beta value!"
    # convert from standart format into matrix
    current_position = sp.Matrix([[i[1] for i in current_position]])
    # θ' = θ - α*▽f(θ)
    next_position = current_position - step_length*position_vector
    # convert from matrix into standard format
    value = [i for i in next_position]
    return [(starting_position[i][0], value[i]) for i in range(len(value))]

def _grad(function, position):
    gradient = []
    for i in range(len(starting_position)):
        gradient.append(function.diff(starting_position[i][0]).subs(position))
    return sp.Matrix([gradient])

def _modulus(position_vector):
    v = 0
    for i in position_vector:
        v += i**2
    return v**0.5


def steepest_gradient():
    current_position = starting_position
    stop = False
    while not stop:
        position_vector = _grad(f, current_position)
        next_position_var = _update_position (-position_vector, current_position)
        next_position_func = f.subs(next_position_var)
        derivative = sp.diff(next_position_func, alpha)
        alpha_value = sp.solve(derivative, alpha)[0]
        next_position = _update_position(-position_vector, current_position, alpha_value)
        print(current_position)
        current_position = next_position
        if _modulus(position_vector) <= 0.005:
            stop = True


    



if __name__ == "__main__":
   steepest_gradient()
