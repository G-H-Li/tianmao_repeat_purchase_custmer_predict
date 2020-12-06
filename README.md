# tianmao_repeat_purchase_custmer_predict

天池新人赛：https://tianchi.aliyun.com/competition/entrance/231576/information

## 项目目的
* 学习相关数据处理知识

## 使用说明
* 从dataset文件夹中读取数据
* 直接运行`purchase_predict.py`,来实现数据预处理、特征提取、模型训练和数据预测

## 文件说明
* `dataset_check.py`：检查数据集的相关内容
* `data_preprocess.py`:进行数据集的预处理
* `feature_image.py`:可视化提取后的特征

## 解题思路
1. 原始数据检查
2. 对无意义，有问题数据进行筛查和修正
3. 从原始数据中初步提取特征，构建训练集（尽量获得有意义且规范化的值
4. 将带标签数据集分为训练集和测试集使用算法进行训练建模
5. 对预测集进行预测，得到结果

## 经验
* 在使用随机森林算法训练后，显然训练结果过拟合，导致最终预测结果准确率较低
