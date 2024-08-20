import csv
import json
import pickle
import timeit

import numpy as np
from hyperopt import STATUS_OK
from hyperopt import tpe, hp, Trials, fmin
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

from cart2pixel import Cart2Pixel, ConvPixel
from deep import CNN2
import matplotlib.pyplot as plt

import time

XGlobal = []
YGlobal = []

XTestGlobal = []
YTestGlobal = []

SavedParameters = []
Mode = ""
Name = ""
best_val_acc = 0

attack_label = 0

# 儲存每一輪結果的變數
results = []

def save_results_to_txt(filename):
    avg_accuracy = np.mean([result['accuracy'] for result in results])
    avg_precision = np.mean([result['precision'] for result in results])
    avg_recall = np.mean([result['recall'] for result in results])
    avg_f1 = np.mean([result['f1'] for result in results])
    
    avg_confusion_matrix = np.mean([result['confusion_matrix'] for result in results], axis=0)
    
    with open(filename, 'w') as file:
        for i, result in enumerate(results):
            file.write(f'{i+1}th results:\n')
            file.write(f'(Accuracy): {result["accuracy"]}\n')
            file.write(f'(Precision): {result["precision"]}\n')
            file.write(f'(Recall): {result["recall"]}\n')
            file.write(f'(F1 Score): {result["f1"]}\n')
            file.write(f'(Confusion Matrix): {result["confusion_matrix"]}\n')
            file.write('\n')
        
        file.write('Average results:\n')
        file.write(f'(Average Accuracy): {avg_accuracy}\n')
        file.write(f'(Average Precision): {avg_precision}\n')
        file.write(f'(Average Recall): {avg_recall}\n')
        file.write(f'(Average F1 Score): {avg_f1}\n')
        file.write(f'(Average Confusion Matrix): \n{avg_confusion_matrix}\n')

def fix(f):
    a = f["TN_val"]
    b = f["FP_val"]
    c = f["FN_val"]
    d = f["TP_val"]
    f["TN_val"] = d
    f["TP_val"] = a
    f["FP_val"] = c
    f["FN_val"] = b
    return f

def fix_test(f):
    a = f["TN_test"]
    b = f["FP_test"]
    c = f["FN_test"]
    d = f["TP_test"]
    f["TN_test"] = d
    f["TP_test"] = a
    f["FP_test"] = c
    f["FN_test"] = b
    return f

def res(cm, val):
    tp = cm[1][1]  # attacks true
    fn = cm[1][0]  # attacks predict normal
    fp = cm[0][1]  # normal predict attacks
    tn = cm[0][0]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    print(f"攻擊樣本總數 (attacks): {attacks}")
    print(f"正常樣本總數 (normals): {normals}")

    if attacks <= normals:
        print("樣本分佈正常 (Sample distribution is normal)")
    elif not val:
        print("錯誤: 攻擊樣本數量大於正常樣本數量 (Error: attacks > normals and val is False)")
        return False, [None] * 7
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / 2
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = [OA, AA, P, R, F1, FAR, TPR]
    return True, r

# hyperopt function to optimize
def hyperopt_fcn(params):
    global SavedParameters
    start_time = time.time()
    print("開始訓練 (start train)")
  
    model, val = CNN2(XGlobal, YGlobal, params)
    train_end_time = time.time()
    print("開始預測 (start predict)")

    predict_start_time = time.time()
    y_predicted = model.predict(XTestGlobal, verbose=0, use_multiprocessing=True, workers=12)
    predict_end_time = time.time()
    
    y_predicted = np.argmax(y_predicted, axis=1)
    elapsed_time = time.time() - start_time
    
    # 打印測試集中的攻擊和正常數量
    test_attack_count = (YTestGlobal == attack_label).sum()
    test_normal_count = (YTestGlobal != attack_label).sum()
    print(f"測試集中的攻擊樣本數量 (Test attack sample count): {test_attack_count}")
    print(f"測試集中的正常樣本數量 (Test normal sample count): {test_normal_count}")
    
    # 打印訓練時間和測試時間
    train_time = train_end_time - start_time
    predict_time = predict_end_time - predict_start_time
    print(f"訓練時間 (Train time): {time.strftime('%H:%M:%S', time.gmtime(train_time))}")
    print(f"測試時間 (Test time): {time.strftime('%H:%M:%S', time.gmtime(predict_time))}")

    cf = confusion_matrix(YTestGlobal, y_predicted)
    print(f"混淆矩陣 (Confusion Matrix): \n{cf}")
    print(f"測試集的 F1_score (test F1_score): {f1_score(YTestGlobal, y_predicted)}")

    # 計算並儲存每一輪的結果
    TP = cf[1][1]
    FN = cf[1][0]
    FP = cf[0][1]
    TN = cf[0][0]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    results.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cf  # 儲存混淆矩陣
    })
    
    K.clear_session()
    SavedParameters.append(val)
    global best_val_acc
    print(f"驗證集準確率 (val acc): {val['F1_score_val']}")

    SavedParameters[-1].update(
    {"balanced_accuracy_test": balanced_accuracy_score(YTestGlobal, y_predicted) * 100, "TP_test": cf[0][0],
        "FN_test": cf[0][1], "FP_test": cf[1][0], "TN_test": cf[1][1], "kernel": params["kernel"],
        "learning_rate": params["learning_rate"],
        "batch": params["batch"],
        "dropout1": params["dropout1"],
        "dropout2": params["dropout2"],
        "time": time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
        "train_time": time.strftime('%H:%M:%S', time.gmtime(train_time)),
        "predict_time": time.strftime('%H:%M:%S', time.gmtime(predict_time))})
    
    cm_val = [[SavedParameters[-1]["TP_val"], SavedParameters[-1]["FN_val"]],
              [SavedParameters[-1]["FP_val"], SavedParameters[-1]["TN_val"]]]

    print(f"驗證集混淆矩陣 (Validation Confusion Matrix): \n{cm_val}")
    r = res(cm_val, True)
    if not r[0]:
        print(f"驗證失敗，結果為 (Validation failed with result): {r}")
        return {'loss': np.inf, 'status': STATUS_OK}
    
    SavedParameters[-1].update({
        "OA_val": r[1][0],
        "P_val": r[1][2],
        "R_val": r[1][3],
        "F1_val": r[1][4],
        "FAR_val": r[1][5],
        "TPR_val": r[1][6]
    })
    cm_test = [[SavedParameters[-1]["TP_test"], SavedParameters[-1]["FN_test"]],
               [SavedParameters[-1]["FP_test"], SavedParameters[-1]["TN_test"]]]

    print(f"測試集混淆矩陣 (Test Confusion Matrix): \n{cm_test}")
    r = res(cm_test, False)
    if not r[0]:
        print(f"測試失敗，結果為 (Test failed with result): {r}")
        return {'loss': np.inf, 'status': STATUS_OK}
    
    SavedParameters[-1].update({
        "OA_test": r[1][0],
        "P_test": r[1][2],
        "R_test": r[1][3],
        "F1_test": r[1][4],
        "FAR_test": r[1][5],
        "TPR_test": r[1][6]
    })

    # Save model
    if SavedParameters[-1]["F1_val"] > best_val_acc:
        print(f"新保存的模型 (new saved model): {SavedParameters[-1]}")
        model.save(Name.replace(".csv", "_model.h5"))
        best_val_acc = SavedParameters[-1]["F1_val"]

    SavedParameters = sorted(SavedParameters, key=lambda i: i['F1_val'], reverse=True)

    try:
        with open(Name, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")
    return {'loss': -val["F1_val"], 'status': STATUS_OK}

def train_norm(param, dataset, norm):
    np.random.seed(param["seed"])
    print("建模資料集 (modelling dataset)")
    global YGlobal
    YGlobal = to_categorical(dataset["Classification"])
    del dataset["Classification"]
    global YTestGlobal
    YTestGlobal = to_categorical(dataset["Ytest"])
    del dataset["Ytest"]

    global XGlobal
    global XTestGlobal

    if not param["LoadFromPickle"]:
        # norm
        Out = {}
        if norm:
            print('進行 Min-Max 正規化 (NORM Min-Max)')
            Out["Max"] = float(dataset["Xtrain"].max().max())
            Out["Min"] = float(dataset["Xtrain"].min().min())
            # NORM
            dataset["Xtrain"] = (dataset["Xtrain"] - Out["Min"]) / (Out["Max"] - Out["Min"])
            dataset["Xtrain"] = dataset["Xtrain"].fillna(0)

        print("轉置資料集 (trasposing)")

        q = {"data": np.array(dataset["Xtrain"].values).transpose(),
             "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": np.argmax(YGlobal, axis=1)}
        print(f"最大A大小 (max_A_size): {q['max_A_size']}")
        print(f"最大B大小 (max_B_size): {q['max_B_size']}")

        # generate images
        XGlobal, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], params=param)

        del q["data"]
        print("生成訓練圖像完成 (Train Images done!)")
        # generate testing set image
        
        dataset["Xtest"] = dataset["Xtest"].drop(dataset["Xtest"].columns[toDelete], axis=1)

        x = image_model["xp"]
        y = image_model["yp"]
        col = dataset["Xtest"].columns
        # col = col.delete(0)
        # print(col)
        # coordinate model
        coor_model = {"coord": ["xp: " + str(i) + "," "yp :" + str(z) + ":" + col for i, z, col in zip(x, y, col)]}
        j = json.dumps(coor_model)
        f = open(param["dir"] + "MI_model.json", "w")
        f.write(j)
        f.close()

        dataset["Xtest"] = np.array(dataset["Xtest"]).transpose()
        print("生成測試圖像 (generating Test Images)")
        print(f"測試集形狀 (Test dataset shape): {dataset['Xtest'].shape}")

        # 確保 XTestGlobal 的形狀一致
        XTestGlobal = []
        for i in range(0, dataset["Xtest"].shape[1]):
            img = ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                            image_model["A"], image_model["B"])
            if img.shape == (image_model["A"], image_model["B"]):
                XTestGlobal.append(img)

        print("生成測試圖像完成 (Test Images done!)")

        # saving testing set
        name = "_" + str(int(q["max_A_size"])) + "x" + str(int(q["max_B_size"]))

        name = name + "_MI"

        filename = param["dir"] + "test" + name + ".pickle"
        f_file = open(filename, 'wb')
        pickle.dump(XTestGlobal, f_file)
        f_file.close()
    else:
        XGlobal = dataset["Xtrain"]
        XTestGlobal = dataset["Xtest"]
    # GAN
    del dataset["Xtrain"]
    del dataset["Xtest"]
    XTestGlobal = np.array(XTestGlobal)
    image_size1, image_size2 = XTestGlobal[0].shape
    XTestGlobal = np.reshape(XTestGlobal, [-1, image_size1, image_size2, 1])
    YTestGlobal = np.argmax(YTestGlobal, axis=1)

    hyperparameters_to_optimize = {"kernel": hp.choice("kernel", np.arange(2, 4 + 1)),
                                    "batch": hp.choice("batch", [16, 32, 64, 128, 256]),
                                    'dropout1': hp.uniform("dropout1", 0, 1),
                                    'dropout2': hp.uniform("dropout2", 0, 1),
                                    "learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),
                                    "epoch": param["epoch"]}

    # output name
    global attack_label
    attack_label = param["attack_label"]

    global Mode
    Mode = "CNN2"

    global Name
    Name = param["dir"] + "res_" + str(int(param["Max_A_Size"])) + "x" + str(int(param["Max_B_Size"]))

    Name = Name + "_MI"

    Name = Name + "_" + Mode + ".csv"
    trials = Trials()
    fmin(hyperopt_fcn, hyperparameters_to_optimize, trials=trials, algo=tpe.suggest, max_evals=param["hyper_opt_evals"])

    print("完成 (done)")
    
    import os
    # 儲存每一筆結果和平均結果
    save_results_to_txt(os.path.join("GAN","dataset", "nids2017result", f"results_{param['enhanced_dataset']}_{param['hyper_opt_evals']}_{param['epoch']}.txt"))

    return 1
