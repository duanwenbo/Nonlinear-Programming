import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb

################### Hyper parameters ##############################
SEED = 1205
BETA1 = 0.8
BETA2 = 0.999
ALPHA = 0.02
EPSILON = 1e-8
iteration = 1000
####################################################################

wandb.init()

def read_data(path):
    df = pd.read_csv(path)
    train_label = _scaling(df.iloc[:-50,-1])
    train_set = _scaling(df.iloc[:-50,:-1])
    verify_set =  _scaling(df.iloc[-50:,:-1])
    verify_label = _scaling(df.iloc[-50:,-1])
    return train_set, train_label, verify_set, verify_label

def visualize_data(train_set, labels):
    """visualize the original data, it is used for selecting the training sets"""
    nrows, ncol = (train_set.shape[1]//3 +1), 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncol, figsize=(18,20))
    for i in range(train_set.shape[1]):
        independent_var = train_set.iloc[:,i]
        ax[i%nrows, i%ncol].plot(independent_var, labels, 'r.', )
        ax[i%nrows, i%ncol].set_title(independent_var.name)
    plt.show()
    # Choose RM and LSTAT as indepedent variables according to the output graph

def _scaling(data):
    """feature scaling, normalize data fo better optimizing performance"""
    min = data.min()
    max = data.max()
    data = data - min
    data = data / (max - min)
    return data

def _perdict_func(current_position, train_set):
    """Linear Regression Perdiction"""
    # f(θ_0, θ_1, .., θ_n) = θ_0 + θ_1 * x_1 + θ_2 * x_2 + ... + θ_n * x_n
    # parameters: 1x(N+1) matrix; train_set: MxN matrix
    constant = np.ones(train_set.shape[0])
    train_set = np.insert(train_set,0, values=constant, axis=1)
    # Mx(N+1) · (N+1)x1  -->  Mx1 matrix
    perdiction = np.dot(train_set, np.transpose(current_position))
    return perdiction

def _compute_grad(current_position, x, y):
    """ compute the gradient of MSE loss func """
    grad = np.empty_like(current_position)
    for i in range(grad.shape[1]):
        if i == 0:
            # partial derivative of  θ_0
            grad[0][i] = ((_perdict_func(current_position, x) - y).sum()) / y.shape[0]
        else:
            # partial derivative of the rest of parameters
            x_i = np.expand_dims(x[:,i-1], axis=1)  # reshape to Mx1
            grad[0][i] = (((_perdict_func(current_position, x) - y)*x_i).sum()) / y.shape[0]
    return grad # 1x(N+1) matrix

def _compute_loss(current_position, train_set, label):
    """mean square error loss between perdict prices and labeled prices"""
    loss = ((_perdict_func(current_position, train_set) - label)**2).sum() / label.shape[0]
    return loss

def evaluate(current_position, train_set, label):
    """visualize the final comparsion"""
    # ideal one
    x = np.arange(0,label.shape[0],1)
    perdict = _perdict_func(current_position, train_set)
    plt.plot(x, perdict)
    plt.plot(x, label)
    plt.legend(labels=['predict price','test set price'])
    plt.xlabel('index of the case')
    plt.ylabel('normalized house price')
    plt.show()

def adam_optimize():
    """Adaptive Moment Estimation"""
    path = ".\\datasets\\house_pricing.csv"
    train_set, label, verify_set, verify_label = read_data(path)
    # visualize_data(train_set, label)
    # select feature parameters
    train_set = pd.concat([train_set['RM'], train_set['LSTAT']], axis=1).to_numpy()
    verify_set = pd.concat([verify_set['RM'], verify_set['LSTAT']], axis=1).to_numpy()
    # train_set = train_set.to_numpy()
    # train_set = np.expand_dims(train_set['RM'], axis=1)
    label = label.to_numpy()
    label = np.expand_dims(label, axis=1)
    verify_label = verify_label.to_numpy()
    verify_label = np.expand_dims(verify_label, axis=1)
    # plt.plot(train_set, label, 'r.')
    # initialize the optimie position
    # The dimention of position is generated according to the number of chosen independ variable
    np.random.seed(SEED)
    current_position = [np.random.random()]*(train_set.shape[1]+1)  # N+1
    # current_position = [0.5,10.,1]
    current_position = np.array([current_position]) # reshape to (N+1)x1 matrix
    #inital first and second moments
    m = v = np.array([[0]*current_position.shape[1]])  # 1xN matrix
    # m, v = [0]*current_position.shape[1], [0]*current_position.shape[1]
    # start optimizing
    for t in range(iteration):
        grad = _compute_grad(current_position, train_set, label)
        new_m = BETA1*m + (1-BETA1)*grad
        new_v = BETA2*v + (1-BETA2)*grad**2
        m_hat = new_m / (1-BETA1**(t+1))
        v_hat = new_v / (1-BETA2**(t+1))
        # update position
        current_position = current_position -ALPHA*m_hat / (v_hat**0.5 + EPSILON)
        # store the current moments for the next iteration
        m = new_m
        v = new_v
        loss = _compute_loss(current_position, train_set, label)
        print("iteration", t, 'loss', loss, "current position: {}".format(current_position))
        wandb.log({"loss":loss})
        
    evaluate(current_position, verify_set, verify_label)

    # perdict_price = _perdict_func(current_position, train_set)
    # plt.plot(train_set, perdict_price)
    # plt.show()

if __name__ == "__main__":
    adam_optimize()