# 参考论文：Fully-Connected Spatial-Temporal Graph Neural Network for Multivariate Time-Series Data (https://arxiv.org/pdf/2309.05305.pdf). 

By: Yucheng Wang, [Yuecong Xu](https://xuyu0010.github.io/), [Jianfei Yang](https://marsyang.site/), [Min Wu](https://sites.google.com/site/wumincf/), [Xiaoli Li](https://personal.ntu.edu.sg/xlli/), [Lihua Xie](https://personal.ntu.edu.sg/elhxie/), [Zhenghua Chen](https://zhenghuantu.github.io/)


# Dataset

Three datasets to evaluate the method, including C-MAPSS, UCI-HAR, and ISRUC-S3.

## C-MAPSS

Access [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/), and put the downloaded dataset into directory 'CMAPSSData'.

For running the experiments on C-MAPSS, directly run main_RUL.py

## UCI-HAR

Access [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones), and put the downloaded dataset into directory 'HAR'.

For running the experiments on UCI-HAR, you need to first run preprocess_UCI_HAR.py to pre-process the dataset. After that, run main_HAR.py

## ISRUC-S3
 
Access [here](https://sleeptight.isr.uc.pt/), and download S3 and put the downloaded dataset into directory 'ISRUC'.

For running the experiments on ISRUC, you need to first run preprocess_ISRUC.py to pre-process the dataset. After that, run main_ISRUC.py
