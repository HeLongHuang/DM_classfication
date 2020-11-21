import Process_data
import torchvision
import torch.nn as nn
import torch.functional as F
import pandas as pd
import numpy as np
import NetWork
import torch
import matplotlib.pyplot as plt
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
pd.set_option('display.max_columns', 100000000)
pd.set_option('display.width', 100000000)
pd.set_option('display.max_colwidth', 10000000)
train_data = Process_data.process_data().get_train_data_csv()
val_data = train_data.iloc[6000:len(train_data)]
train_data = train_data.iloc[0:6000]  #训练
test_data = Process_data.process_data().get_test_data_csv()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = NetWork.NetWork()
network.to(device)

optimizer = torch.optim.SGD(network.parameters(),lr = 0.001,momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()


x = []
for i in range(100):
    total_loss = 0
    for j in range(len(train_data)):
        t1 = time.time()
        temp_array = train_data.iloc[j]  # 读到一行数据
        train_array = []
        for k in temp_array:
            train_array.append(k)
        label = torch.tensor([temp_array[-1]],dtype=torch.long).to(device)
        train_array.pop()
        data = torch.tensor([train_array]).to(device)
        prediction = network(data).to(device)
        loss = loss_func(prediction,label).to(device)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        if j % 100 == 0:
            print("epoch:{0:>3}  step:{1:>4} loss:{2:>20} time:{3:>20}".format(str(i),str(j),str(loss.item()),t2 - t1))
    x.append(total_loss)
torch.save(network.state_dict(), "model.pth")

plt.plot(x)
plt.show()


