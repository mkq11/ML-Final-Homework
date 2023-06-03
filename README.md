## 项目说明
本项目包含了数据挖掘导论期末作业的部分内容，目前能够建立基础的神经网络。为提高最终得分，所有代码仅使用numpy。模型已使用pytorch验证，精度能达到99%，参数量大约为300k。

为达到最终效果，还需要实现：
* 相关网络结构
  * Conv2d
  * BatchNorm2d
  * AdaptiveAvgPool2d
* 图像变换相关函数
  * Pad
  * RandomRotation
  * RandomCrop

本项目的实现过程中参考了pytorch的用法。

## 作业要求
1. 数字识别机器学习：做0～9数字的识别问题（数据文件名：mnist_data.csv），综合考虑特征选择、特征提取、分类器的设计等问题，给出一个你认为最好的问题解决方案，提交内容包括实现程序、大作业论文。
2. 可组队协作探索，每队上限2人，以组为单位提交，需要提交论文和实现程序，其中论文需要各自写。
3. 课程论文提交时间： 2023年6月18日晚23:59之前，以收到的时间为准。
4. 课程论文成绩：40%[论文报告、算法实现]