from numpy import *

def compute_error_for_given_point(b, m, points):
        totalError = 0
        for i in range(len(points)):
            x = points[i,0]
            y = points[i,1]
            totalError += (y - (m*x + b))**2
        return totalError/float(len(points))
        

def step_grandient(b_current, m_current, points, learning_rate):
    #gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += (2/N) * (((m_current * x) + b_current) - y)
        m_gradient += (2/N) * x * (((m_current * x) + b_current) - y)
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, initial_b, initial_m, n_interations, learning_rate):
    b = initial_b
    m = initial_m

    for i in range(n_interations):
        b,m = step_grandient(b,m,array(points),learning_rate)
    return [b,m]

def run():
    points = genfromtxt('data.csv',delimiter=',')
    #hiperparemeter
    learning_rate = 0.0001
    #y = mx + b
    initial_b = 0
    initial_m = 0
    n_interations = 1000
    [b,m] = gradient_descent_runner(points, initial_b, initial_m, n_interations, learning_rate)
    print(b)
    print(m)


if __name__ == '__main__':
    run()