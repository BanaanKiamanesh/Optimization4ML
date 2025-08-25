from numpy import array, zeros_like, sqrt


# Basic Gradient Descent
def gradient_descent(grad_func, start, lr=0.01, steps=100):
    w = array(start, dtype=float)
    
    trajectory = [w.copy()]
    for _ in range(steps):
        grad = grad_func(w[0], w[1])
        w -= lr * grad
        trajectory.append(w.copy())
    
    return array(trajectory)

# RMSProp Optimizer
def rmsprop(grad_func, start, lr=0.01, steps=100, beta=0.9, epsilon=1e-8):
    w = array(start, dtype=float)
    s = array(zeros_like(w), dtype=float)
    
    trajectory = [w.copy()]
    for _ in range(steps):
        grad = grad_func(w[0], w[1])
        s = beta * s + (1 - beta) * (grad ** 2)
        w -= lr * grad / (sqrt(s) + epsilon)
        trajectory.append(w.copy())
    
    return array(trajectory)

# Adam Optimizer
def adam(grad_func, start, lr=0.01, steps=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
    w = array(start, dtype=float)
    m = array(zeros_like(w), dtype=float)
    v = array(zeros_like(w), dtype=float)
    
    trajectory = [w.copy()]
    for t in range(1, steps + 1):
        grad = grad_func(w[0], w[1])
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        w -= lr * m_hat / (sqrt(v_hat) + epsilon)
        trajectory.append(w.copy())
    
    return array(trajectory)

# Momentum Based Gradient Descent
def momentum_gd(grad_func, start, lr=0.01, steps=100, momentum=0.9):
    w = array(start, dtype=float)
    v = array(zeros_like(w), dtype=float)
    
    trajectory = [w.copy()]
    for _ in range(steps):
        grad = grad_func(w[0], w[1])
        v = momentum * v - lr * grad
        w += v
        trajectory.append(w.copy())
    
    return array(trajectory)