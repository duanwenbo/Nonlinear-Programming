import os
import sympy as sp  # tools for matrix, calculus operation
from sympy.parsing.sympy_parser import parse_expr


class Optimizer:
    def __init__(self, objective_function, initial_position, type) -> None:
        self.obj_expression = objective_function
        self.start_position_expression = initial_position
        self.type = type
        self.obj = ""
        self.start_position = ""
        self.GOLDEN_RATIO = 0.618
        self.EPSILON = 0.005
        self.alpha = sp.symbols('alpha')
        self._parsing_input()

    def conjugate_gradient(self):
        print("Start optimizing...")
        _detect_previous_doc()
        # first round cycle
        current_position = self.start_position
        gradient = self._grad(self.obj, current_position)
        position_vector = gradient  # when j=1
        next_position_var = self._update_position(position_vector,
                                                  current_position)
        obj_func = self.obj.subs(next_position_var)
        step_length = abs(self._golden_search(obj_func))
        next_position = self._update_position(position_vector,
                                              current_position, step_length)

        _record(current_position, position_vector, 0, step_length, True)
        current_position = next_position

        stop = False
        while not stop:
            new_gradient = self._grad(self.obj, current_position)
            beta = self._HS_form(new_gradient, gradient)  # searching beta
            new_position_vector = new_gradient - beta * position_vector
            next_position_var = self._update_position(new_position_vector,
                                                      current_position)
            obj_func = self.obj.subs(next_position_var)
            step_length = abs(self._golden_search(obj_func))  # searching alpha
            next_position = self._update_position(new_position_vector,
                                                  current_position,
                                                  step_length)

            _record(current_position, new_position_vector, beta, step_length)
            # update parameters
            current_position = next_position
            gradient = new_gradient
            position_vector = new_position_vector
            if self._modulus(position_vector) < self.EPSILON:
                stop = True
                current_position = [(i[0], "%.2f" % i[1])
                                    for i in current_position]
                print("The optimal point is: {}".format(current_position))
                with open("solution.txt", "a+") as file:
                    file.write("The optimal point is {}".format(current_position))

    def _parsing_input(self):
        self.obj = parse_expr(self.obj_expression)
        # format text input
        limits = self.start_position_expression.replace(" ", "").split(",")
        limits = [tuple(i.split("=")) for i in limits]
        # extract variables from the expression
        symbols = list(self.obj.free_symbols)
        var_list = []
        for i, constrain in enumerate(limits):
            for element in symbols:
                if str(element) == constrain[0]:
                    var_list.append((element, limits[i][1]))
        self.start_position = var_list

    def _update_position(self,
                         position_vector,
                         current_position,
                         step_length=0):
        if step_length == 0:
            step_length = self.alpha
        assert len(self.start_position) == sp.shape(
            position_vector)[1], "check beta value!"
        # convert from standart format into matrix
        current_position = sp.Matrix([[i[1] for i in current_position]])
        # θ' = θ + α*▽f(θ)
        if self.type == "maximisation":
            next_position = current_position + step_length * position_vector
        elif self.type == "minimisation":
            next_position = current_position - step_length * position_vector
        else:
            raise AttributeError
        # convert from matrix into standard format
        value = [i for i in next_position]
        result = [(self.start_position[i][0], (value[i]))
                  for i in range(len(value))]
        return result

    def _grad(self, function, position):
        gradient = []
        for i in range(len(self.start_position)):
            gradient.append(
                function.diff(self.start_position[i][0]).subs(position))
        return sp.Matrix([gradient])

    def _modulus(self, position_vector):
        v = 0
        for i in position_vector:
            v += i**2
        return v**0.5

    def _HS_form(self, current_gradint, previous_gradient):
        # reshape matrix before calculating
        current_gradint, previous_gradient = current_gradint.T, previous_gradient.T
        # Hestenes-Stiefel form
        numerator = current_gradint.T * (current_gradint - previous_gradient)
        denominator = previous_gradient.T * previous_gradient
        return numerator * denominator.inv()[0]

    def _golden_search(self, func):
        # find the optimal step length
        a, b = -99., 99.
        difference = abs(a - b)
        while difference > self.EPSILON:
            distance = self.GOLDEN_RATIO * (b - a)
            m, n = a + distance, b - distance
            fm, fn = func.subs(self.alpha, m), func.subs(self.alpha, n)
            if self.type == "maximisation":
                if fm < fn:  # > minimisation < maximisation
                    b = m
                else:
                    a = n
            elif self.type == "minimisation":
                if fm > fn:  # > minimisation < maximisation
                    b = m
                else:
                    a = n
            else:
                raise AttributeError
            difference = abs(a - b)
        opt_point = (b + a) / 2
        return opt_point


def _detect_previous_doc():
    if os.path.exists("log.txt"):
        os.remove("log.txt")
    if os.path.exists("solution.txt"):
        os.remove("solution.txt")


def _record(position, direction_vector, beta, step_length, initial=False):
    # record intermediate steps
    a = position
    position = [(i[0], "%.2f" % float(i[1])) for i in position]
    position = [str(i) for i in position]
    if initial:
        current_position = "initial position: {} ".format(" ,".join(position))
        beta = sp.Matrix([[0]])
    else:
        current_position = "current position: {} ".format(" ,".join(position))
    direction = str(["%.2f" % i for i in list(direction_vector)]) + "^T"

    with open('log.txt', 'a+') as f:
        f.write("{}\n".format(current_position))
        f.write("next update direction vector: {}\n".format(direction))
        f.write("where direction updating parameter beta: %.2f\n" % beta[0])
        f.write("step length parameter alpha:%.2f\n\n" % step_length)


if __name__ == "__main__":
    ######################## input the NLP question there ########################
    opt = Optimizer(
        objective_function="(x_1 - 1)**2 + (x_2 -2)**2 + (x_3 - 3)**2",
        initial_position="x_1=0.5, x_2=0.5, x_3 = 0.5",
        type="minimisation")
    ##############################################################################
    opt.conjugate_gradient()
