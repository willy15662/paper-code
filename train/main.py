import os
import pickle
import pandas as pd
import csv
import numpy as np
import train
import time
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Parameters
param = {"Max_A_Size": 15 ,"Max_B_Size": 15, "ValidRatio": 0.1, "seed": 200,
         "dir": "GAN/dataset/dataset1/",
         "LoadFromPickle": False, "rate":1,  "hyper_opt_evals": 15, "epoch": 50,  
         "enhanced_dataset": "gan"   #gan, None
         }
# Start timez
start_time = time.time()

dataset = 1  # change dataset
if dataset == 1:
    train_file = 'train15_allxy.csv'
    test_file = 'test15_allxy.csv'
    classif_label = 'attack_cat'
    param["attack_label"] = 1
elif dataset == 2:
    train_file = 'train99_5_xy.csv'
    test_file = 'test99_xy.csv'
    classif_label = 'label'
    param["attack_label"] = 1
elif dataset == 3:
    train_file = 'train_xy17.csv'
    test_file = 'test_xy17.csv'
    classif_label = 'Label'
    param["attack_label"] = 1

if not param["LoadFromPickle"]:
    data = {}
    with open(param["dir"] + train_file, 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
        data["Classification"] = data["Xtrain"][classif_label]
        del data["Xtrain"][classif_label]

        # 打印修改前的攻擊和正常數量
        original_attack_count = (data["Classification"] == param["attack_label"]).sum()
        original_normal_count = (data["Classification"] != param["attack_label"]).sum()
        print(f"原始攻擊樣本數量: {original_attack_count}")
        print(f"原始正常樣本數量: {original_normal_count}")
        
        # 直接儲存 y_train
        with open(param["dir"] + 'y_train.pickle', 'wb') as f:
            pickle.dump(data["Classification"], f)
            
    with open(param["dir"] + test_file, 'r') as file:
        Xtest = pd.DataFrame(list(csv.DictReader(file)))
        print(Xtest.shape)
        Xtest.replace("", np.nan, inplace=True)
        Xtest.dropna(inplace=True)
        data["Xtest"] = Xtest.astype(float)
        data["Ytest"] = data["Xtest"][classif_label]
        del data["Xtest"][classif_label]

        # 直接儲存 y_test
        with open(param["dir"] + 'y_test.pickle', 'wb') as f:
            pickle.dump(data["Ytest"], f)

    model = train.train_norm(param, data, norm=False)

else:
    images = {}

    method = 'MI'
    if param["enhanced_dataset"] == "gan":
        f_myfile = open(param["dir"] + 'XTrain50A%.pickle', 'rb')
        images["Xtrain"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'YTrain50A%.pickle', 'rb')
        images["Classification"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'test_' + str(param['Max_A_Size']) + 'x' + str(
            param['Max_B_Size']) + '_' + method + '.pickle', 'rb')
        images["Xtest"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'y_test.pickle', 'rb')
        images["Ytest"] = pickle.load(f_myfile)
        f_myfile.close()
    else:
        f_myfile = open(param["dir"] + 'train_' + str(param['Max_A_Size']) + 'x' + str(
            param['Max_B_Size']) + '_' + method + '.pickle', 'rb')
        images["Xtrain"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'y_train.pickle', 'rb')
        images["Classification"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(
            param["dir"] + 'test_' + str(param['Max_A_Size']) + 'x' + str(
                param['Max_B_Size']) + '_' + method + '.pickle',
            'rb')
        images["Xtest"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'y_test.pickle', 'rb')
        images["Ytest"] = pickle.load(f_myfile)
        f_myfile.close()
        
    model = train.train_norm(param, images, norm=False)
    print("train_norm 執行完成")

# End time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time // 3600:.0f}h {elapsed_time % 3600 // 60:.0f}m {elapsed_time % 60:.0f}s")
