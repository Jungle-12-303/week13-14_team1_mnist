import numpy as np

def xavier_init(fan_in, fan_out):
    """
    Xavier (Glorot) Initialization
    Sigmoid, Tanh 활성화 함수에 최적화된 초기화 방식.
    
    Args:
        fan_in: 입력 노드 수
        fan_out: 출력 노드 수
    Returns:
        (fan_in, fan_out) 형태의 가중치 행렬
    """
    # 표준편차: sqrt(1 / fan_in)
    std = np.sqrt(1.0 / fan_in)
    return std * np.random.randn(fan_in, fan_out)

def he_init(fan_in, fan_out):
    """
    He (Kaiming) Initialization
    ReLU 계열 활성화 함수에 최적화된 초기화 방식.
    
    Args:
        fan_in: 입력 노드 수
        fan_out: 출력 노드 수
    Returns:
        (fan_in, fan_out) 형태의 가중치 행렬
    """
    # 표준편차: sqrt(2 / fan_in)
    std = np.sqrt(2.0 / fan_in)
    return std * np.random.randn(fan_in, fan_out)