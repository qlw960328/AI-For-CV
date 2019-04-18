#### linear regression
###############################
import numpy as np
import random

# define hypothesis function
def inference(w, b, x):
    pred_y = w * x + b
    return pred_y

# define cost function
def loss_function(w, b, x_list, real_y_list):
    aver_loss = 0.0

    for i in range(len(x_list)):
        aver_loss = aver_loss + 0.5 * (inference(w, b, x_list[i]) - real_y_list[i]) ** 2
    aver_loss = aver_loss / len(real_y_list)
    return aver_loss

# define difference
def gradient(pred_y, real_y, x):
    diff = pred_y - real_y
    dw = diff * x
    db = diff
    return dw, db

# define gradient descent step
# main function: calculate w and b
def step_gradient(batch_x_list, batch_real_y_list, lr, w, b):
    aver_w, aver_b = 0, 0
    batch_size = len(batch_x_list)
    for j in range(batch_size):
        pred_y = inference(w, b, batch_x_list[j])
        dw, db = gradient(pred_y, batch_real_y_list[j], batch_x_list[j])
        aver_w = aver_w + dw
        aver_b = aver_b + db
    aver_w = aver_w / len(batch_x_list)
    aver_b = aver_b / len(batch_x_list)
    w = w - lr * aver_w
    b = b - lr * aver_b
    return w, b

# define train function
def train(x_list, real_y_list, lr, batch_size, max_iter):
    w = 0
    b = 0
    for i in range(max_iter):
        batch_idex = np.random.choice(len(x_list), batch_size)
        batch_x_list = [x_list[i] for i in batch_idex]
        batch_real_y_list = [real_y_list[i] for i in batch_idex]
        w, b = step_gradient(batch_x_list, batch_real_y_list, lr, w, b)
        loss = loss_function(w, b, batch_x_list, batch_real_y_list)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss:{0}'.format(loss))

def gen_sample_data():
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()
    num_sample = 100
    x_list = []
    real_y_list = []
    for j in range(num_sample):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.randint(-1, 1)
        x_list.append(x)
        real_y_list.append(y)
    return x_list, real_y_list, w, b

def run():
    x_list, real_y_list, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_list, real_y_list, lr, 50, max_iter)

if __name__ == '__main__':
    run()
