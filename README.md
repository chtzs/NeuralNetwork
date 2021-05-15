# NeuralNetwork
简单的三层神经网络引擎。

## 参考
全部代码参考自《Python 神经网络编程》，这确实是一本不可多得的好书。

## 用法
1. 下载训练数据[mnist_train.csv](https://pjreddie.com/media/files/mnist_train.csv)和测试数据[mnist_test.csv](https://pjreddie.com/media/files/mnist_test.csv)放入trains_data文件夹中。由于文件过大，故不放到github上。
2. 直接运行，由于我已经训练好了一个模型module.npz和它的备份my_train_module.npz，你可以直接用这个模型测试数据。
3. 查看输出结果。我自己跑的模型（200层隐藏结点，0.2的学习率）可以达到95.4%的正确率。
4. （可选）调用`test_img`对自己绘制的任意28*28大小的白底黑字数字图片进行测试。
