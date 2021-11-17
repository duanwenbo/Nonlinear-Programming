import sympy as sp  # tools for matrix, calculus operation

################ edit aera ########################
# TODO: mattrix format of input obj func
"""set the objective function there"""
# 1. declare variables
x_1, x_2, x_3 = sp.symbols('x_1 x_2 x_3') 
# 2. initialize obj func
f = (x_1 - 1)**2 + (x_2 -2)**2 + (x_3 - 3)**2
# 3. declare init position
starting_position = [(x_1,0.5), (x_2,0.5), (x_3,0.5)]  # standard format of position
#####################################################
alpha = sp.symbols('alpha')  # declare step lenghth var

EPSILON = 0.005  # tolerance
GOLDEN_RATIO = 0.618



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

def _HS_form(current_gradint, previous_gradient):
    # reshape matrix before calculating
    current_gradint, previous_gradient = current_gradint.T, previous_gradient.T
    # Hestenes-Stiefel form
    numerator = current_gradint.T*(current_gradint-previous_gradient)
    denominator = previous_gradient.T*previous_gradient
    return numerator*denominator.inv()[0]

def golden_search(func):
    # find the optimal step length
    a,b = -99.,99. 
    opt_point = (b+a) / 2
    gradient = func.diff(alpha).subs([(alpha,opt_point)])
    while abs(gradient) > EPSILON:
        distance = GOLDEN_RATIO * (b-a)
        m,n = a+distance, b-distance
        fm, fn = func.subs(alpha,m), func.subs(alpha,n)
        if fm > fn:
            b = m
        elif fm < fn:
            a = n
        else:
            break
        opt_point = (b+a) / 2
    return opt_point

def conjugate_gradient():
    # first round cycle
    current_position = starting_position
    gradient= _grad(f, current_position)
    position_vector = - gradient  # when j=1
    next_position_var = _update_position(position_vector, current_position)
    next_positon_func = f.subs(next_position_var)
    step_length = golden_search(next_positon_func)
    next_position = _update_position(position_vector, current_position, step_length)
    current_position = next_position

    stop = False
    while not stop:
        new_gradient = _grad(f, current_position)
        beta = _HS_form(new_gradient, gradient)  # searching beta
        new_position_vector = - new_gradient + beta * position_vector
        next_position_var = _update_position(new_position_vector, current_position)
        next_positon_func = f.subs(next_position_var)
        step_length = golden_search(next_positon_func)  # searching alpha
        next_position = _update_position(position_vector, current_position, step_length)
        # update parameters
        current_position = next_position
        gradient = new_gradient
        position_vector = new_position_vector
        if _modulus(position_vector) < EPSILON:
            stop = True
            print(current_position)

def steepest_gradient():
    current_position = starting_position
    stop = False
    while not stop:
        position_vector = _grad(f, current_position)
        next_position_var = _update_position (-position_vector, current_position)
        next_position_func = f.subs(next_position_var)
        derivative = sp.diff(next_position_func, alpha)
        step_len = sp.solve(derivative, alpha)[0]
        next_position = _update_position(-position_vector, current_position, step_len)
        print(current_position)
        current_position = next_position
        if _modulus(position_vector) <= EPSILON:
            stop = True
            print(current_position)


if __name__ == "__main__":
   steepest_gradient()
