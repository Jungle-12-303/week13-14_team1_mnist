# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU
from layers import Affine, BatchNorm, Dropout
from losses import SoftmaxWithLoss, softmax


class MultiLayerNetExtend:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        """
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        self.params = {}
        self.grads = {}
        self.layers = OrderedDict()
        self.last_layer = SoftmaxWithLoss()

        sizes = [784, 512, 256, 10]
        for idx in range(1, len(sizes)):
            weight_key = f"W{idx}"
            bias_key = f"b{idx}"
            self.params[weight_key] = (
                np.random.randn(sizes[idx - 1], sizes[idx])
                * np.sqrt(2.0 / sizes[idx - 1])
            )
            self.params[bias_key] = np.zeros(sizes[idx])

            self.layers[f"Affine{idx}"] = Affine(
                self.params[weight_key],
                self.params[bias_key],
            )

            is_hidden_layer = idx < len(sizes) - 1
            if is_hidden_layer:
                if use_batchnorm:
                    gamma_key = f"gamma{idx}"
                    beta_key = f"beta{idx}"
                    self.params[gamma_key] = np.ones(sizes[idx])
                    self.params[beta_key] = np.zeros(sizes[idx])
                    self.layers[f"BatchNorm{idx}"] = BatchNorm(
                        self.params[gamma_key],
                        self.params[beta_key],
                    )

                self.layers[f"ReLU{idx}"] = ReLU()

                if use_dropout:
                    self.layers[f"Dropout{idx}"] = Dropout(dropout_ratio)

        self.grads = {key: np.zeros_like(value) for key, value in self.params.items()}

    def predict(self, x, train=False):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 score
        """
        out = x
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                out = layer.forward(out, train=train)
            else:
                out = layer.forward(out)
        return out

    def forward(self, x, train=True):
        """테스트와 노트북 호환을 위해 score를 확률로 변환해 반환합니다."""
        scores = self.predict(x, train=train)
        return softmax(scores)

    def loss(self, x, y):
        """현재 모델의 score를 만든 뒤 SoftmaxWithLoss로 loss를 계산합니다."""
        scores = self.predict(x, train=True)
        return self.last_layer.forward(scores, y)

    def backward(self, dout=1):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        if np.isscalar(dout):
            dout = self.last_layer.backward(dout)

        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        for name, layer in self.layers.items():
            if isinstance(layer, Affine):
                layer_idx = name.replace("Affine", "")
                self.grads[f"W{layer_idx}"] = layer.dW
                self.grads[f"b{layer_idx}"] = layer.db
            elif isinstance(layer, BatchNorm):
                layer_idx = name.replace("BatchNorm", "")
                self.grads[f"gamma{layer_idx}"] = layer.dgamma
                self.grads[f"beta{layer_idx}"] = layer.dbeta

        return self.grads

    def gradient(self, x, y):
        """책의 MultiLayerNetExtend.gradient() 흐름처럼 loss 계산 후 역전파합니다."""
        self.loss(x, y)
        return self.backward()


class NeuralNetwork(MultiLayerNetExtend):
    """과제 템플릿에서 사용하는 이름. 구현은 MultiLayerNetExtend 스타일을 따릅니다."""
