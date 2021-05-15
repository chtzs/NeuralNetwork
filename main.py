from typing import List, Tuple

from neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plot
import imageio
import os
import time


def load_trains_data(filename: str, max_line: int = 1000):
    # 训练数据
    _trains_data: List[List[float]] = []
    # 期望数据
    _targets_list: List[Tuple[int, List[float]]] = []
    data_list = []
    with open(filename) as f:
        i = 0
        while True and i < max_line:
            line = f.readline()
            if line == "":
                break
            data_list.append(line)
            i += 1

    for data in data_list:
        if data.strip() == "":
            continue
        all_values = data.split(",")
        # 读入期望数据
        target = [0.01 for _ in range(0, 10)]
        target[int(all_values[0])] = 0.99
        _targets_list.append((int(all_values[0]), target))
        # 读入训练数据
        _trains_data.append([float(x) for x in all_values[1:]])

    return _trains_data, _targets_list


def fix_data(data: List[float]):
    return [x / 255.0 * 0.99 + 0.01 for x in data]


def show(_trains_data: List[float], _targets_list: List[float], result: np.ndarray):
    i = 0
    for i in range(0, 10):
        if _targets_list[i] == 0.99:
            break
    print("ans: ", i)
    i = 0
    for j in range(0, 10):
        if result[j] > result[i]:
            i = j
    print("predict: ", i)
    plot.imshow(np.asfarray(_trains_data).reshape(28, 28), cmap="Greys")
    plot.show()


def test(n: NeuralNetwork):
    print("reading testing data...")
    test_data, test_target = load_trains_data("./trains_data/mnist_test.csv", 10000)
    print("start querying...")
    score = 0
    for i in range(0, len(test_data)):
        result = n.query(test_data[i])[1]
        recognize = np.argmax(result)
        if int(recognize) == test_target[i][0]:
            score += 1

    print("total: ", len(test_data), ", correct: ", score, ", 正确率: ", score / len(test_data))


def test_img(n: NeuralNetwork, img_path: str) -> None:
    img_arr = imageio.imread(img_path, as_gray=True)
    img_data = 255.0 - img_arr.reshape(784)
    result = n.query(fix_data(img_data))[1]
    print("recognize: ", np.argmax(result))

    plot.imshow(img_data.reshape(28, 28), cmap="Greys")
    plot.show()


def train_nn(n: NeuralNetwork):
    t = time.perf_counter()
    print("start reading...")
    trains_data, targets_list = load_trains_data("./trains_data/mnist_train.csv", 60000)
    print("start learning...")
    epochs = 4
    for e in range(epochs):
        print("start generation ", e)
        for i in range(0, len(trains_data)):
            n.train(fix_data(trains_data[i]), targets_list[i][1])

    print(f'training coast: {time.perf_counter() - t:.8f}s')


if __name__ == '__main__':
    # 输入、隐藏和输出结点的数量
    input_nodes = 28 * 28
    hidden_nodes = 100
    output_nodes = 10
    # 学习率
    learning_rate = 0.2
    # 神经学习引擎
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # 训练模型
    path = "./trains_data/module"
    if os.path.exists(path + ".npz"):
        print("load from file?")
        if input().strip().lower() == 'y':
            n.load_from_file(path + ".npz")
        else:
            os.remove(path + ".npz")
            train_nn(n)
            n.save_to_file(path)
    else:
        train_nn(n)
        n.save_to_file(path)

    print("start testing...")
    # test(n)
    test_img(n, "./trains_data/8.png")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
