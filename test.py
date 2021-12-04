from sympy.parsing.sympy_parser import parse_expr

expression = "(x_1 -1)**2 + (x_2 -2)**2 +(x_3 -3)**2"
limits = "x_1 = 0, x_2 = 2, x_3 = 5"



def generate_limit_expression(limits, expression):
    # format text input
    limits = limits.replace(" ","").split(",")
    limits = [tuple(i.split("=")) for i in limits]
    # extract variables from the expression
    symbols = list(parse_expr(expression).free_symbols)
    var_list = []
    for i, constrain in enumerate(limits):
        for element in symbols:
            if str(element) == constrain[0]:
                var_list.append((element, limits[i][1]))
    return var_list

a = generate_limit_expression(limits, expression)

a = ["a=1", "b=2"]
# b = [tuple(i.split("=")) for i in a]
# print(b)