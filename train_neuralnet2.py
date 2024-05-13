# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
#from two_layer_net import TwoLayerNet
from six_layer_net import sixLayerNet
import matplotlib.pyplot as plt
from common.util import smooth_curve

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = sixLayerNet(input_size=784, hidden_size1=15, hidden_size2=14, hidden_size3=13, hidden_size4 = 12, hidden_size5 = 11, output_size=10)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 480 
learning_rate = 0.019

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    grad = network.gradient(x_batch, t_batch)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5', 'W6', 'b6'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc:", train_acc, "test acc:", test_acc)

max_iterations = len(train_loss_list)
x = np.arange(max_iterations)

plt.plot(x, smooth_curve(train_loss_list))
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 6)
plt.show()
