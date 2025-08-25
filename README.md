# Optimization4ML

A Python project demonstrating various optimization algorithms and techniques commonly used in machine learning. This repository contains implementations of different optimization methods that can be applied to machine learning problems.

## Purpose

This project is created **only for demonstration and learning purposes**. It serves as an educational resource for understanding how different optimization algorithms work, their implementation details, and their applications in machine learning contexts. The code is designed to be clear and readable rather than production-ready.

## Mathematical Formulations

### Gradient Descent
**Update Rule:**
$$w_{t+1} = w_t - \alpha \nabla f(w_t)$$

### RMSProp
**Momentum:**
$$s_t = \beta s_{t-1} + (1-\beta)(\nabla f(w_t))^2$$

**Update Rule:**
$$w_{t+1} = w_t - \frac{\alpha}{\sqrt{s_t} + \epsilon} \nabla f(w_t)$$

### Adam
**First Moment (Momentum):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla f(w_t)$$

**Second Moment (Velocity):**
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla f(w_t))^2$$

**Bias Correction:**
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Update Rule:**
$$w_{t+1} = w_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### Momentum Gradient Descent
**Velocity Update:**
$$v_t = \mu v_{t-1} - \alpha \nabla f(w_t)$$

**Parameter Update:**
$$w_{t+1} = w_t + v_t$$

Where:
- $w_t$: parameters at step $t$
- $\alpha$: learning rate
- $\nabla f(w_t)$: gradient of the objective function
- $\beta, \beta_1, \beta_2$: momentum parameters
- $\mu$: momentum coefficient
- $\epsilon$: small constant for numerical stability
