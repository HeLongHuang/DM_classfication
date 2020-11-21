import NetWork
import torch
import pandas as pd
import os
import Process_data


os.environ['KMP_DUPLICATE_LIB_OK']='True'   # 画图bug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pd.set_option('display.max_columns', 100000000)
pd.set_option('display.width', 100000000)
pd.set_option('display.max_colwidth', 10000000)
train_data = Process_data.process_data().get_train_data_csv()
val_data = train_data.iloc[6000:len(train_data)]
train_data = train_data.iloc[0:6000]  # 训练
test_data = Process_data.process_data().get_test_data_csv()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = NetWork.NetWork()
network.load_state_dict(torch.load("model.pth"))
network.eval()
network.to(device)

total_correct = 0
for i in range(len(val_data)):
    temp_array = val_data.iloc[i]
    val_array = []
    for j in temp_array:
        val_array.append(j)
    label = torch.tensor(temp_array[-1],dtype=torch.float32).to(device)
    val_array.pop()
    data = torch.tensor([val_array]).to(device)
    output = network(data).to(device)
    index = output.argmax()
    if index == label:
        total_correct = total_correct + 1
    print("Validating:{0:>4} acc:{1:>5}".format(i,"true" if index == label else "false"))


print("acc：",total_correct / len(val_data))