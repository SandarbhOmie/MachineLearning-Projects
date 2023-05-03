
import numpy as np
# Gradient Descent for linear Regression: we need to find the value of w and b which will generate our data
# Linea reg eq  => yhat = wx+b
# loss = (y-yhat)**2/N

# parameters for calculation
x = np.random.randn(10,1)
y = 2*x + np.random.rand()

# Parameters
w = 0.0
b = 0.0
# Hyperparameters : using here to estimate our model performance
learning_rate = 0.01
print(x.shape[0])
# function for Gradient Descent
def descend(x, y, w, b, learning_rate): 
    dldw = 0.0 
    dldb = 0.0 
    N = x.shape[0] 

# loss  = (y-(wx=b))**2
    for xi,yi in zip(x,y):
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))

    w = w-learning_rate*(1/N)*dldw
    b = b- learning_rate*(1/N)*dldb

    return w,b




#interatively
for epoch in range(400):
    w,b=descend(x,y,w,b,learning_rate)
    yhat = w*x + b
    loss =np.divide(np.sum((y-yhat)**2, axis =0),x.shape[0])
    print(f'{epoch} loss is {loss}, parameters w: {w},b:{b}')
