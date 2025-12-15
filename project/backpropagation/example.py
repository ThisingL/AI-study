import numpy as np

# Sigmoid 激活函数及其导数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# 初始化参数
def initialize_parameters(input_size, hidden_size, output_size):
    """
    随机初始化权重和偏置
    返回: parameters 字典
    """
    np.random.seed(42)  # 固定随机种子，便于复现
    
    parameters = {
        'W1': np.random.randn(hidden_size, input_size) * 0.5,
        'b1': np.zeros((hidden_size, 1)),
        'W2': np.random.randn(output_size, hidden_size) * 0.5,
        'b2': np.zeros((output_size, 1))
    }
    return parameters

# 前向传播
def forward_propagation(X, parameters):
    """
    TODO: 实现前向传播
    
    输入:
        X: 输入数据，形状 (2, m)，m 是样本数
        parameters: 包含 W1, b1, W2, b2
    
    返回:
        A2: 输出层激活值
        cache: 包含中间值的字典 (Z1, A1, Z2, A2)，反向传播时需要
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # TODO: 实现这里
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

# 计算损失
def compute_cost(A2, Y):
    """
    计算均方误差
    """
    m = Y.shape[1]
    cost = np.sum((A2 - Y) ** 2) / m
    return cost

# 反向传播
def backward_propagation(X, Y, parameters, cache):
    """
    TODO: 实现反向传播
    
    输入:
        X: 输入数据
        Y: 真实标签
        parameters: 参数字典
        cache: 前向传播的中间值
    
    返回:
        grads: 包含所有梯度的字典
    """
    m = X.shape[1]
    W2 = parameters['W2']
    
    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']
    
    # TODO: 实现反向传播
    delta2 = 2 * (A2 - Y) * sigmoid_derivative(Z2)
    dW2 = delta2 @ A1.T
    db2 = np.sum(delta2, axis=1, keepdims=True)
    delta1 = (W2.T @ delta2) * sigmoid_derivative(Z1)  # 修复：W2.T 在前
    dW1 = delta1 @ X.T
    db1 = np.sum(delta1, axis=1, keepdims=True)
    
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

# 参数更新
def update_parameters(parameters, grads, learning_rate):
    """
    TODO: 使用梯度下降更新参数
    """
    parameters['W1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']
    parameters['W2'] -= learning_rate * grads['dW2']
    parameters['b2'] -= learning_rate * grads['db2']
    
    return parameters

# 训练模型
def train(X, Y, hidden_size, learning_rate, num_iterations):
    """
    训练神经网络
    """
    input_size = X.shape[0]
    output_size = Y.shape[0]
    
    parameters = initialize_parameters(input_size, hidden_size, output_size)
    
    for i in range(num_iterations):
        # 前向传播
        A2, cache = forward_propagation(X, parameters)
        
        # 计算损失
        cost = compute_cost(A2, Y)
        
        # 反向传播
        grads = backward_propagation(X, Y, parameters, cache)
        
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # 每 1000 次迭代打印损失
        if i % 1000 == 0:
            print(f"Iteration {i}: Cost = {cost:.6f}")
    
    return parameters

# XOR 数据
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

# 训练
parameters = train(X, Y, hidden_size=2, learning_rate=10.0, num_iterations=10000)

# 测试
A2, _ = forward_propagation(X, parameters)


print("\n预测结果:")
print("输入:", X.T)
print("预测:", A2.T)
print("真实:", Y.T)