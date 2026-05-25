# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9

    
    y_pred = np.array([
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.70, 0.05, 0.04, 0.03, 0.03],
        [0.80, 0.02, 0.01, 0.01, 0.03, 0.02, 0.04, 0.03, 0.02, 0.02],
        [0.05, 0.10, 0.60, 0.05, 0.04, 0.03, 0.04, 0.03, 0.03, 0.03],
    ])

    y_true = np.array([2, 0])
    0번째 샘플 정답은 2번 클래스
    1번째 샘플 정답은 0번 클래스
    """
    # TODO: 정답 클래스 확률의 log 값을 이용해 batch 평균 cross entropy를 계산하세요.
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.
    y_pred = np.clip(y_pred, 1e-7, None)
    batch_size = y_pred.shape[0]
    idx = np.arange(batch_size) # 반복문 처럼 쓰기 위한 범위 값
    correct_probs = y_pred[idx, y_true] # 정답값 확률
    out = -np.average(np.log(correct_probs)) # 손실 함수 식
    return out
    raise NotImplementedError("cross_entropy_loss를 구현하세요.")
