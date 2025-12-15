import numpy as np
import matplotlib.pyplot as plt

# 复制之前的函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def initialize_parameters(input_size, hidden_size, output_size, seed):
    np.random.seed(seed)
    parameters = {
        'W1': np.random.randn(hidden_size, input_size) * 0.5,
        'b1': np.zeros((hidden_size, 1)),
        'W2': np.random.randn(output_size, hidden_size) * 0.5,
        'b2': np.zeros((output_size, 1))
    }
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = np.sum((A2 - Y) ** 2) / m
    return cost

def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]
    W2 = parameters['W2']

    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']

    delta2 = 2 * (A2 - Y) * sigmoid_derivative(Z2)
    dW2 = delta2 @ A1.T
    db2 = np.sum(delta2, axis=1, keepdims=True)
    delta1 = (W2.T @ delta2) * sigmoid_derivative(Z1)
    dW1 = delta1 @ X.T
    db1 = np.sum(delta1, axis=1, keepdims=True)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

def update_parameters(parameters, grads, learning_rate):
    parameters['W1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']
    parameters['W2'] -= learning_rate * grads['dW2']
    parameters['b2'] -= learning_rate * grads['db2']
    return parameters

def train(X, Y, hidden_size, learning_rate, num_iterations, seed):
    """训练并返回损失历史"""
    input_size = X.shape[0]
    output_size = Y.shape[0]

    parameters = initialize_parameters(input_size, hidden_size, output_size, seed)
    cost_history = []

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        cost_history.append(cost)
        grads = backward_propagation(X, Y, parameters, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

    return parameters, cost_history

# XOR 数据
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

# 测试不同的学习率
learning_rates = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
num_iterations = 5000

plt.figure(figsize=(15, 10))

for idx, lr in enumerate(learning_rates):
    print(f"\n测试学习率: {lr}")

    # 使用相同的随机种子保证公平比较
    parameters, cost_history = train(X, Y, hidden_size=4,
                                     learning_rate=lr,
                                     num_iterations=num_iterations,
                                     seed=42)

    # 测试最终结果
    A2, _ = forward_propagation(X, parameters)
    final_cost = cost_history[-1]

    print(f"最终损失: {final_cost:.6f}")
    print(f"预测: {A2.T.flatten()}")

    # 绘制损失曲线
    plt.subplot(2, 3, idx + 1)
    plt.plot(cost_history)
    plt.title(f'学习率 = {lr}')
    plt.xlabel('迭代次数')
    plt.ylabel('损失 (Cost)')
    plt.yscale('log')  # 使用对数坐标，更清楚
    plt.grid(True, alpha=0.3)

    # 标注最终损失
    plt.text(0.5, 0.95, f'最终损失: {final_cost:.6f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/thisingl/Downloads/AI-study/project/backpropagation/learning_rate_comparison.png', dpi=150)
print("\n图表已保存到: learning_rate_comparison.png")
print("\n观察要点:")
print("1. 学习率太小(0.1)：收敛很慢")
print("2. 学习率适中(0.5-2.0)：快速且稳定收敛")
print("3. 学习率太大(5.0-10.0)：可能震荡或发散")
