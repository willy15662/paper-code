# paper-code
Multi-Layer Edge Computing and IoT Intrusion Detection System with Integrated GAN and XAI

Dataset
You can download the pre-processed datasets from our Google Drive:

Download Dataset

The dataset is divided into two main folders:

ACGAN Folder - Contains the GAN model and related files:

acgan.py: Script to train the ACGAN model.
gan.py: Contains the generator and discriminator code.
generator.py: Generates data using the trained model and adds it to the original dataset.
Train Folder - Contains training data and data transformation scripts:

Data_preprocessing.ipynb: Pre-processes the dataset and splits it into training and testing sets.
cart2pixel.py: Converts network traffic data into image data.
deep.py: CNN model implementation.
main.py: Main script to select datasets and call other scripts.
train.py: Executes the training of the CNN model.
part2.ipynb: Training script for the second part of the IoT model.
Usage Instructions
Data Preprocessing:

Run the Data_preprocessing.ipynb script to preprocess the dataset and split it into training and testing sets.
Main Script Configuration:

Open main.py.
Adjust the parameters in the param section:
Max_A_Size: Image size.
dir: Directory path of the dataset.
LoadFromPickle: Set to False to convert the dataset to image format.
rate: Adjust the attack ratio.
hyper_opt_evals: Number of training iterations.
epoch: Number of epochs per training session.
enhanced_dataset: Set to False initially.
Convert to Image Data:

Set LoadFromPickle to False and run the script to convert the data into image format.
Train GAN Model:

Run acgan.py to train the GAN model using the image data.
Generate Additional Data:

Run generator.py to generate new data using the trained GAN model and append it to the original dataset.
Train with GAN-Enhanced Data:

Reopen main.py, set LoadFromPickle to True, and enhanced_dataset to gan.
Run the script to train and test the model with the GAN-enhanced dataset.
Train IoT Model:

Execute part2_nb15.ipynb to train the second part of the IoT intrusion detection model.




切分好的資料集檔案可以在 https://drive.google.com/drive/folders/1GcJl2iBii8gel5FoR1powVxvAs-dwG7Q?usp=sharing 雲端硬碟中下載

檔案分為兩部分
ACGAN資料夾為生成資料部分包含3個檔案
1.acgan.py : 用來訓練ACGAN模型
2.gan.py : 模型的生成器和判別器程式
3.generator.py : 使用訓練完的模型生成資料並加到原資料中

train資料夾為訓練資料和資料轉換部分，共6個檔案
1.Data_preprocessing.ipynb : 將資料進行前處理，並切割成訓練集和測試集
2.cart2pixel.py : 將流量資料轉為圖像資料
3.deep.py : CNN模型程式碼
4.main.py : 主程式，選擇資料集並呼叫其他程式
5.train.py : 執行CNN模型的訓練
6.part2.ipynb : 第二部分IoT端訓練程式碼

使用步驟
1.先用Data_preprocessing檔案將資料進行前處理，並切割成訓練集和測試集
2.打開main.py檔案
3.更改param中的參數，Max_A_Size:圖像大小，dir:資料夾路徑，LoadFromPickle:資料是否已轉為圖像資料
  rate:調整攻擊佔比，hyper_opt_evals:訓練次數，epoch:每次訓練回合數，enhanced_dataset:是否使用加入gan的資料
4.將LoadFromPickle改為False，將檔案轉為圖像
5.執行acgan.py使用轉好的圖像資料訓練生成模型
6.執行generator.py生成資料並加到原始資料中
7.執行main.py並將LoadFromPickle改為True，enhanced_dataset改為gan，測試加入生成資料後的訓練結果
8.執行part2_nb15.ipynb訓練第二部分模型

