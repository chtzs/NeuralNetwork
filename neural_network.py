import numpy
import numpy as np
from scipy.special import expit as sigmoid
from typing import List, Tuple


class NeuralNetwork:
    """
    简单的三层神经网络
    分别是输入层，隐藏层，输出层
    其中输入层的作用仅仅是输入，也就是说此神经网络实际起作用的只有两层（隐藏层和输出层）
    """

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float):
        """
        对神经网络初始化
        :param input_nodes: 输入结点的数量
        :param hidden_nodes: 隐藏层结点的数量
        :param output_nodes: 输出层结点的数量
        :param learning_rate: 学习率，也就是梯度下降法的步进比例
        """
        self.input_count = input_nodes
        self.hidden_count = hidden_nodes
        self.output_count = output_nodes
        self.learning_rate = learning_rate
        # 学习权重
        # self.weight_input_hidden = np.random.rand(self.hidden, self.inputs) - 0.5
        # self.weight_hidden_output = np.random.rand(self.hidden, self.outputs) - 0.5
        # 输入到隐藏层的权重
        # 正态分布取随机数，标准差为hidden_count开根号的倒数，最后一个参数是随机出来的数组大小
        self.weight_input_hidden: np.ndarray = np.random.normal(0.0, pow(self.hidden_count, -0.5),
                                                                (self.hidden_count, self.input_count))
        # 隐藏层到输出层的权重
        self.weight_hidden_output: np.ndarray = np.random.normal(0.0, pow(self.output_count, -0.5),
                                                                 (self.output_count, self.hidden_count))
        # 设置激活函数
        self.activation_function = lambda x: sigmoid(x)

    def train(self, inputs_list: List[float], targets_list: List[float]) -> None:
        """
        训练数据，即查询并更新权重
        :param inputs_list: 数据列表，大小必须是inputs_nodes
        :param targets_list: 目标数据，用于纠错和自我学习
        :return: 无
        """
        # ---part1 计算数据与误差---
        # 把targets_list转换成numpy的array类型
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 计算当前数据
        hidden_outputs, final_outputs = self.query(inputs_list)
        # 计算误差
        output_errors = targets - final_outputs

        # ---part2 根据误差更新权重---
        # 把误差输出反向传播到隐藏层，通过权重合理分配误差项
        hidden_errors = np.dot(self.weight_hidden_output.T, output_errors)
        # 梯度下降法更新权重
        # 更新隐藏层到输出层的权重
        self.weight_hidden_output += self.learning_rate * np.dot(output_errors * final_outputs * (1 - final_outputs),
                                                                 hidden_outputs.T)
        # 更新输入层到隐藏层的权重
        self.weight_input_hidden += self.learning_rate * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs),
                                                                inputs.T)

    def query(self, inputs_list: List[float]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        查询函数，将一组输入转化成三层神经网络的输出
        :param inputs_list: 输入，是数组类型，数量要和input_nodes匹配
        :return: 元组，第一个参数是隐藏层的输出，第二个参数是最终层的输出
        """
        # 把inputs_list转换成numpy的array类型
        inputs = np.array(inputs_list, ndmin=2).T
        # 输入层的输出到隐藏层的输入的转换（直接和权重矩阵相乘）
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        # 隐藏层的输出（用sigmoid函数转化输入）
        hidden_outputs = self.activation_function(hidden_inputs)
        # 最终输出层的输入
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        # 输出层的输出
        final_outputs = self.activation_function(final_inputs)
        return hidden_outputs, final_outputs

    def save_to_file(self, file_path: str) -> None:
        np.savez(file_path, self.weight_input_hidden, self.weight_hidden_output)

    def load_from_file(self, file_path: str) -> None:
        arr = np.load(file_path)
        self.weight_input_hidden = arr["arr_0"]
        self.weight_hidden_output = arr["arr_1"]
