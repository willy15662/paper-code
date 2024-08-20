# Multi-Layer Edge Computing and IoT Intrusion Detection System with Integrated GAN and XAI

This repository contains the code and data for our project on intrusion detection using GAN and XAI in a multi-layer edge computing and IoT environment.

## Dataset

You can download the pre-processed datasets from our Google Drive:

[Download Dataset](https://drive.google.com/drive/folders/1GcJl2iBii8gel5FoR1powVxvAs-dwG7Q?usp=sharing)

The dataset is divided into two main folders:

1. **ACGAN Folder** - Contains the GAN model and related files:
   - **acgan.py**: Script to train the ACGAN model.
   - **gan.py**: Contains the generator and discriminator code.
   - **generator.py**: Generates data using the trained model and adds it to the original dataset.

2. **Train Folder** - Contains training data and data transformation scripts:
   - **Data_preprocessing.ipynb**: Pre-processes the dataset and splits it into training and testing sets.
   - **cart2pixel.py**: Converts network traffic data into image data.
   - **deep.py**: CNN model implementation.
   - **main.py**: Main script to select datasets and call other scripts.
   - **train.py**: Executes the training of the CNN model.
   - **part2.ipynb**: Training script for the second part of the IoT model.

## Usage Instructions

1. **Data Preprocessing**: 
   - Run the `Data_preprocessing.ipynb` script to preprocess the dataset and split it into training and testing sets.

2. **Main Script Configuration**:
   - Open `main.py`.
   - Adjust the parameters in the `param` section:
     - `Max_A_Size`: Image size.
     - `dir`: Directory path of the dataset.
     - `LoadFromPickle`: Set to `False` to convert the dataset to image format.
     - `rate`: Adjust the attack ratio.
     - `hyper_opt_evals`: Number of training iterations.
     - `epoch`: Number of epochs per training session.
     - `enhanced_dataset`: Set to `False` initially.

3. **Convert to Image Data**:
   - Set `LoadFromPickle` to `False` and run the script to convert the data into image format.

4. **Train GAN Model**:
   - Run `acgan.py` to train the GAN model using the image data.

5. **Generate Additional Data**:
   - Run `generator.py` to generate new data using the trained GAN model and append it to the original dataset.

6. **Train with GAN-Enhanced Data**:
   - Reopen `main.py`, set `LoadFromPickle` to `True`, and `enhanced_dataset` to `gan`.
   - Run the script to train and test the model with the GAN-enhanced dataset.

7. **Train IoT Model**:
   - Execute `part2_nb15.ipynb` to train the second part of the IoT intrusion detection model.
