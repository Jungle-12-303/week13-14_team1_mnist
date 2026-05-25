# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss

class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(self, initializer, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        """
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        # TODO: params dict를 만들고 Affine/BatchNorm/ReLU/Dropout layer를 순서대로 구성하세요.
        # 권장 구조: 784 -> 512 -> 256 -> 10
        # self.layers는 OrderedDict로 만들고, self.grads는 params와 같은 key를 갖게 합니다.
        self.layers = OrderedDict()
        self.params = {}
        self.grads = {}

        hidden_sizes = [512, 256]
        prev_size = 784
        
        for idx, hidden_size in enumerate(hidden_sizes, start=1):
            self._add_hidden_layer(
                layer_idx=idx,
                input_size=prev_size,
                output_size=hidden_size,
                initializer=initializer,
                use_batchnorm=use_batchnorm,
                use_dropout=use_dropout,
                dropout_ratio=dropout_ratio
            )
            prev_size = hidden_size

        self._add_output_layer(
            layer_idx=len(hidden_sizes) + 1,
            input_size=prev_size,
            output_size=10,
            initializer=initializer
        )

        self.last_layer = Softmax()


        #raise NotImplementedError("NeuralNetwork.__init__을 구현하세요.")

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        # TODO: self.layers를 순서대로 통과시키고 마지막에 Softmax를 적용하세요.
        for layer in self.layers.values():
            x = layer.forward(x)

        return self.last_layer.forward(x)
        # raise NotImplementedError("NeuralNetwork.forward를 구현하세요.")

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        # TODO: layer를 역순으로 통과시키고 Affine/BatchNorm의 gradient를 self.grads에 모으세요.
        for key, layer in reversed(self.layers.items()):
            dout = layer.backward(dout)

            if "Affine" in key:
                idx = key.replace("Affine", "")
                self.grads['W' + idx] = layer.dW
                self.grads['b' + idx] = layer.db
            elif "BatchNorm" in key:
                idx = key.replace("BatchNorm", "")
                self.grads['gamma' + idx] = layer.dgamma
                self.grads['beta' + idx] = layer.dbeta

        return dout

        # raise NotImplementedError("NeuralNetwork.backward를 구현하세요.")

    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)


    def _add_hidden_layer(self, layer_idx, input_size, output_size, initializer,
                        use_batchnorm=False, use_dropout=False, dropout_ratio=0.5):
        """
        Affine -> BatchNorm(optional) -> ReLU -> Dropout(optional)
        형태의 hidden layer를 self.params와 self.layers에 추가한다.
        """

        # W, b 생성
        self.params[f"W{layer_idx}"] = initializer(input_size, output_size)
        self.params[f"b{layer_idx}"] = np.zeros(output_size)

        # Affine
        self.layers[f"Affine{layer_idx}"] = Affine(
            self.params[f"W{layer_idx}"],
            self.params[f"b{layer_idx}"]
        )

        # BatchNorm
        if use_batchnorm:
            self.params[f"gamma{layer_idx}"] = np.ones(output_size)
            self.params[f"beta{layer_idx}"] = np.zeros(output_size)

            self.layers[f"BatchNorm{layer_idx}"] = BatchNorm(
                self.params[f"gamma{layer_idx}"],
                self.params[f"beta{layer_idx}"]
            )

        # ReLU
        self.layers[f"ReLU{layer_idx}"] = ReLU()

        # Dropout
        if use_dropout:
            self.layers[f"Dropout{layer_idx}"] = Dropout(dropout_ratio)

    def _add_output_layer(self, layer_idx, input_size, output_size, initializer):
        """
        출력층 Affine layer를 추가한다.
        """

        self.params[f"W{layer_idx}"] = initializer(input_size, output_size)
        self.params[f"b{layer_idx}"] = np.zeros(output_size)

        self.layers[f"Affine{layer_idx}"] = Affine(
            self.params[f"W{layer_idx}"],
            self.params[f"b{layer_idx}"]
    )