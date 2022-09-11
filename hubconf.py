dependencies = ['torch']

import torch
import numpy as np
import math

dtype = torch.float
device = torch.device("cpu")    

def load_data():

    print ('loading data...')
    
    x = torch.linspace(0, 1, 100, device=device, dtype=dtype)
    y = 3 + 4 * x + 5 * x**2 + 6 * x**3
        
    return x, y
    
    
def polyfit_autograd(x,y):

    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-4

    # 2022 Aug 23 - Kalidas (ykalidas@iittp.ac.in)
    optimizer = torch.optim.SGD([a,b,c,d], lr=learning_rate, momentum=0)
    
    for t in range(10000):
    
        # Forward pass: compute predicted y using operations on Tensors.
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()

        # 2022 Aug 23 - Kalidas (ykalidas@iittp.ac.in)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 1000 == 99:
            print(t, loss.item())    
        
    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
        
    return a, b, c, d