from numpy import array

def f(w1, w2):
    return (w1**2 + w2 - 11)**2 + (w1 + w2**2 - 7)**2

# Analytical gradient of f
def grad_f(w1, w2):
    a = w1**2 + w2 - 11
    b = w1 + w2**2 - 7

    return array([4*w1*a + 2*b, 2*a + 4*w2*b], dtype=float)
