# -*- coding: utf-8 -*-
"""파라미터 업데이트 규칙을 모아 둔 optimizer 모듈."""

import numpy as np


class SGD:
    """
    확률적 경사하강법(SGD).

    가장 단순한 optimizer로, 각 파라미터를 gradient 반대 방향으로 lr만큼 이동합니다.
    """

    def __init__(self, lr=0.01):
        """Args: lr: 한 번 업데이트할 때 gradient에 곱할 학습률."""
        self.lr = lr

    def update(self, params: dict, grads: dict):
        """
        params dict의 모든 파라미터를 제자리(in-place)에서 갱신합니다.
        -> return 안해줘도 됨. dict내부 값 알아서 저장됨

        params = {          grads = {
            "W1": ...,          "W1": ..., 
            "b1": ...,          "b1": ...,
            "W2": ...,          "W2": ...,
            "b2": ...,          "b2": ...,
        }                   }

        수식: W - lr * ∂L/∂W
        """
        # TODO: params[key]를 gradient 반대 방향으로 업데이트하세요.
        for key in params.keys():
            # params[key] = params[key] - grads[key]*self.lr 
            # -> 기존 params[key] 버리고 새 배열 만들어서 params[key]에 연결하는 방식 
            # -> 해당 배열을 공유하는 다른 layer에 영향을 줌
            params[key] -= grads[key] * self.lr
        return
        raise NotImplementedError("SGD.update를 구현하세요.")


class Adam:
    """
    Adam Optimizer.

    gradient의 이동평균(m)과 제곱 이동평균(v)을 함께 사용해 파라미터별 학습률을 조절합니다.
    MNIST 과제에서는 SGD보다 빠르게 손실이 내려가는지 비교해 볼 수 있습니다.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """Args: lr: Adam 업데이트의 기본 학습률."""
        self.lr = lr
        self.m, self.v = {}, {}
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def update(self, params: dict, grads: dict):
        """
        Adam 공식에 따라 params dict의 모든 파라미터를 갱신합니다.

        Adam = SGD + grad 평균 m + grad 제곱 평균 v + bias correction
        
        m = β1 * m + (1 - β1) * grad
        v = β2 * v + (1 - β2) * grad²

        bias correction
        m_hat = m / (1 - β1ᵗ)
        v_hat = v / (1 - β2ᵗ)
        
        param = param - lr * m_hat / (sqrt(v_hat) + 1e-7)
        """
        # TODO: m, v 이동평균과 bias correction을 사용해 params를 업데이트하세요.
        self.t = self.t + 1
        for key in params:
            if key not in self.m: # 초기화 과정
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.m[key] * self.beta1 + (1 - self.beta1) * grads[key]
            self.v[key] = self.v[key] * self.beta2 + (1 - self.beta2) * (grads[key]) ** 2

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta1 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)
        
        return
        raise NotImplementedError("Adam.update를 구현하세요.")
