# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def softmax(x):
    """Softmax 함수. 1차원/2차원 입력을 모두 처리합니다."""
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    batch_size = y_pred.shape[0]
    clipped = np.clip(y_pred, 1e-7, 1.0)
    correct_probs = clipped[np.arange(batch_size), y_true]
    return -np.mean(np.log(correct_probs))


class SoftmaxWithLoss:
    """
    Softmax와 Cross Entropy Loss를 하나로 묶은 마지막 layer.

    『밑바닥부터 시작하는 딥러닝』의 구현처럼 forward에서 예측 확률과 정답을
    저장하고, backward에서 (y - t) / batch_size 형태의 gradient를 만듭니다.
    """

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_loss(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]

        if self.t.ndim == 2:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size

        return dx * dout
