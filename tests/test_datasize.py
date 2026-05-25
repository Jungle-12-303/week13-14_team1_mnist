import numpy as np

# 데이터 로드
data = np.load('data/mnist.npz')

# 파일 내부의 배열 이름(key) 확인
print(data.files) # 예: ['x_train', 'y_train', 'x_test', 'y_test']

# 학습 데이터 하나를 추출하여 최대/최소값 확인
x_train = data['x_train']
print(f"Max: {x_train.max()}, Min: {x_train.min()}, Dtype: {x_train.dtype}")