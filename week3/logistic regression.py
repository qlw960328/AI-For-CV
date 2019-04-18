# logistic regression
###############################

import numpy as np
import random
import matplotlib.pyplot as plt

# define hypothesis function
def hypothesis_function(w, b, x):
    pred_y = 1 / (1 + np.exp(-(w * x + b)))
    return pred_y

# defince loss function
def loss_function(x_list, real_y_list, w, b):
    aver_loss = 0.0
    for i in range(len(x_list)):
        pred_y = hypothesis_function(w, b, x_list[i])
        aver_loss = aver_loss + (-real_y_list[i] * np.log(pred_y) - (1-real_y_list[i]) * np.log(1-pred_y))
    aver_loss = aver_loss / len(x_list)
    return aver_loss

# define gradient
def gradient(pred_y, real_y, x):
    diff = pred_y - real_y
    dw = diff * x
    db = diff
    return dw, db

# define one step gradient descent
def step_gradient(batch_x_list, batch_real_y_list, lr, w, b):
    aver_w, aver_b = 0, 0
    for i in range(len(batch_x_list)):
        pred_y = hypothesis_function(w, b, batch_x_list[i])
        dw, db = gradient(pred_y, batch_real_y_list[i], batch_x_list[i])
        aver_w = aver_w + dw
        aver_b = aver_b + db
    aver_w = aver_w / len(batch_x_list)
    aver_b = aver_b / len(batch_real_y_list)
    w = w - lr * aver_w
    b = b - lr * aver_b
    return w, b

# define train process
def train(x_list, real_y_list, batch_size, lr, max_iter):
    w, b = 0, 0
    loss_list = []
    for i in range(max_iter):
        batch_idex = np.random.choice(len(x_list), batch_size)
        batch_x_list = [x_list[i] for i in batch_idex]
        batch_real_y_list = [real_y_list[i] for i in batch_idex]
        w, b = step_gradient(batch_x_list, batch_real_y_list, lr, w, b)
        loss = loss_function(batch_x_list, batch_real_y_list, w, b)
        loss_list.append(loss)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss:{0}'.format(loss))
    return loss_list


# define sample data
def gan_sample_data():
    w = random.randint(0, 15) + random.random()
    b = random.randint(0, 10) + random.random()
    num_sample = 100
    x_list = []
    real_y_list = []
    for i in range(num_sample):
        x = random.randint(0, 100) * random.random()
        y = w * x + b
        y = 1 / (1 + np.exp(-y)) + random.random() * random.randint(-1, 1)
        x_list.append(x)
        real_y_list.append(y)
    return x_list, real_y_list, w, b

# plot loss figure
def plot_fig(loss_list):
    x_axis = range(0, 10000)
    fig = plt.figure()
    fig.set_title = ('loss')
    fig.set_xlabel = ('loss number')
    fig.set_ylabel = ('loss vlue')
    plt.plot(x_axis, loss_list)

    plt.show()


# define run function
def run():
    x_list, real_y_list, w, b = gan_sample_data()
    lr = 0.001
    max_iter = 10000
    loss_list = train(x_list, real_y_list, 50, lr, max_iter)
    plot_fig(loss_list)




if __name__ == '__main__':
    run()