# paper-code
Multi-Layer Edge Computing and IoT Intrusion Detection System with Integrated GAN and XAI

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

