from numpy import *

def f(w1, w2):
    return (w1**2 + w2 - 11)**2 + (w1 + w2**2 - 7)**2

# Analytical gradient of f
def grad_f(w1, w2):
    a = w1**2 + w2 - 11
    b = w1 + w2**2 - 7

    return array([4*w1*a + 2*b, 2*a + 4*w2*b], dtype=float)


# def f(w1, w2):
#     return 0.3 * sin(3 * w1) * sin(3 * w2) + 0.1 * (w1**2 + w2**2)

# # Analytical gradient of f_complex
# def grad_f(w1, w2):
#     df_dw1 = 3 * cos(3 * w1) * sin(3 * w2) + 0.2 * w1
#     df_dw2 = 3 * sin(3 * w1) * cos(3 * w2) + 0.2 * w2
#     return array([df_dw1, df_dw2], dtype=float)