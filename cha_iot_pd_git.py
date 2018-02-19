# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import time
from matplotlib import pyplot as plt

mFile_name="sensors.csv"

# ニューラルネットワーク
class MyChain(Chain):
    def __init__(self, n_units=10):
        super(MyChain, self).__init__(
             l1=L.Linear(1, n_units),
             l2=L.Linear(n_units, n_units),
             l3=L.Linear(n_units, 1))

    def __call__(self, x_data, y_data):
        x = Variable(x_data.astype(np.float32).reshape(len(x_data),1)) # Variableオブジェクトに変換
        y = Variable(y_data.astype(np.float32).reshape(len(y_data),1)) # Variableオブジェクトに変換
        return F.mean_squared_error(self.predict(x), y)

    def  predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        return h3

    def get_predata(self, x):
        return self.predict(Variable(x.astype(np.float32).reshape(len(x),1))).data

# main
if __name__ == "__main__":
    # 学習データ
    rdDim = pd.read_csv("sensors.csv", names=('id', 'temp', 'time') )
    fDim = rdDim["temp"]
    #print(fDim[:10] )
    #quit()
    y_train = np.array(fDim, dtype = np.float32).reshape(len(fDim),1)
    #print(y_train[:20] )
    #quit()
    #print(fDim )
    xDim =np.arange(len(fDim))
    x_train =np.array(xDim, dtype = np.float32).reshape(len(xDim),1)
    #print(y_train )
    #quit()
    # test-data
    n_train = int(len(x_train) * 0.9)
    x_test = x_train[:n_train]
    y_test = y_train[:n_train]
    #print(x_test )
    #quit()
    N= len(x_train)
    N_test  =len(x_test )
    #quit()
    
    # 学習パラメータ
    batchsize = 10
    n_epoch = 500
    n_units = 10
    # モデル作成
    model = MyChain(n_units)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    # 学習ループ
    train_losses =[]
    test_losses =[]
    print ("start...")
    start_time = time.time()
    for epoch in range(1, n_epoch + 1):
        # training
        perm = np.random.permutation(N)
        sum_loss = 0
        for i in range(0, N, batchsize):
            x_batch = x_train[perm[i:i + batchsize]]
            y_batch = y_train[perm[i:i + batchsize]]
            model.zerograds()
            loss = model(x_batch,y_batch)
            sum_loss += loss.data * batchsize
            loss.backward()
            optimizer.update()
        average_loss = sum_loss / N
        train_losses.append(average_loss)
        # test
        loss = model(x_test,y_test)
        test_losses.append(loss.data)
        # 学習過程を出力
        if epoch % 10 == 0:
            print ("epoch: {}/{} train loss: {} test loss: {}".format(epoch, n_epoch, average_loss, loss.data) )
    #
    print ("end" )
    interval = int(time.time() - start_time)
    print ("実行時間: {}sec".format(interval) )
    #quit()
    # # 学習結果のグラフ作成
    # 予測値の取得
    y_pred = model.predict(x_train.astype(np.float32))
    theta = x_train
    #theta =xDim
    test = model.get_predata(theta)
    plt.plot(theta, y_train, label = "temp")
    plt.plot(theta, test, label = "predict")
    plt.legend()
    plt.grid(True)
    plt.title("IoT-data")
    plt.xlabel("theta")
    plt.ylabel("temperature")
    plt.show()

