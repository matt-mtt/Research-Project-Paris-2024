import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

def plot_data(noise, truth):
    plt.figure(figsize=(10,5))
    plt.plot(truth[:,0], truth[:,1], color='r')
    plt.scatter(noise[:,0], noise[:,1], color='b', s=7)
    plt.show()

def plot_pred(x_pred, y_pred, test_loss, truth, mm, percent_train=50):
    # Scaler inverse transform
    y_pred = mm.inverse_transform(y_pred)
    # Plot
    plt.figure(figsize=(10,5)) #plotting
    plt.axvline(x=np.quantile(truth[:,0],percent_train/100), c='r', linestyle='--')
    plt.plot(truth[:,0], truth[:,1], label='Actual Data') #actual plot
    plt.plot(x_pred, y_pred, label='Predicted Data') #predicted plot
    plt.title(f'Visualization of predictions with {percent_train}% of training data.\nMAE on test set : {test_loss:1.5f}')
    plt.legend()
    plt.show() 
