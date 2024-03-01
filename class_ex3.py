
# -*- coding: utf-8 -*-
##################################### square of x
# square of x
def x2(x):
    return x**2


# derivative of x2
def x2_(x):
    # your code here:
    return 2*x

# starting point
X = 10

# your code here:
lr = 0.001
num_of_steps = 1000
for i in range(num_of_steps):
    # your code here:
   X = X - lr * x2_(X)

print(f'Grad_Des_X^2: {X}')
#
# ##################################### x to the power of 4

# x to the power of 4
def x4(x):
    return x**4


# derivative of x4
def x4_(x):
    # your code here:
    return 4*(x**3)


# starting point
X = 10

# your code here:
lr = 0.001
num_of_steps = 10
for i in range(num_of_steps):
     X = X - lr * x4_(X)

print(f'Grad_Des_X^4: {X}')
#
#
# """## - find the minimums of x^2 and x^4 using  Gradient Descent"""

# starting point for the Gradient Descent
X2 = 10
X4 = 10

# your code here:
lr = 0.001
num_of_steps = 100
for i in range(num_of_steps):
    X2 += X2 - lr * x2_(X)
    X4 += X4 - lr * x4_(X)

print("X2:{} \t X4:{} ".format(X2, X4))

# starting point for the Momentum methos
X2m = 10
X4m = 10

"""## - find the minimums of x^2 and x^4 using the Momentum methos """
lrm = 0.001
mu = 0.95
vx2 = 0
vx4 = 0
# your code here:
num_of_steps = 1000000
for i in range(num_of_steps):
    vx2 = mu * vx2 - lrm * x2_(X2m)
    X2m += vx2
    vx4 = mu * vx4 - lrm * x4_(X4m)
    X4m += vx4

print("X2m:{} \t X4m:{}".format(X2m, X4m))


