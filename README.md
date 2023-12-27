# Modified-DGCNN (M-DGCNN)
This repo contain modified version of DGCNN, change the input data from textfile (*.txt) into lasfile using laspy and add compatibility for UTM Point Cloud Data. For complete use and explanation, please check the [original repo](https://github.com/AnTao97/dgcnn.pytorch)

## Requirements
- Python >= 3.7
- PyTorch >= 1.2
- CUDA >= 10.0
- Package: glob, h5py, sklearn, plyfile, torch_scatter

## Data
All point cloud data must placed in `data`, it now accept LAS data format also with UTM compatibility. The code only accept 7 columns of point cloud consists of XYZ RGB + Label.

## How to Run:
1. Place point cloud data (*.LAS) in `data`.
2. Change name of data in `list.txt` and `npy_data_list.txt` is the same with point cloud data in `data`
3. Run data preparation
   ```
   python data_preparation.py
   ```
 4. Run the training script
    ```
    python train_semseg.py --exp_name=exp --test_area=3 --batch_size=16 --test_batch_size=8 --epoch=35
    ```
 5. Training if using pre-trained model
    ```
    python train_semseg.py --exp_name=exp --test_area=3 --batch_size=16 --test_batch_size=8 --epoch=35 --model_root=log/sem_seg/exp/checkpoint/
    ```
 6. Run the evaluation script after training finished
     ```
    python test_semseg.py --batch_size=8 log_dir=exp --test_area=3
    ```
 7. Predict without testing
    ```
    python predict_semseg.py --batch_size=8 log_dir=exp --test_area=3
    ```
##  Update: December 2023
If you want to try the code first, it now available at [Google Colab](https://drive.google.com/drive/folders/1H6OamW16ZWPEEgh5Lo4fEbvVK-rj63h9?usp=sharing)
