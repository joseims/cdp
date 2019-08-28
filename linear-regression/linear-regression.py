from numpy import *
import matplotlib.pyplot as plt

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
        b_gradient += - (2/N) * (y - ((m_current * x) + b_current))
        m_gradient += - (2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, initial_b, initial_m, n_interations, learning_rate):
    b = initial_b
    m = initial_m
    b_ = initial_b[-1]
    m_ = initial_m[-1]
    c = [compute_error_for_given_point(b_,m_,points)]
    error0 = float('inf')
    error1 = 0
    while abs(error1 - error0) > 0.000001:
        b_,m_ = step_grandient(b_,m_,array(points),learning_rate)
        b.append(b_)
        m.append(m_)
        c_ = compute_error_for_given_point(b_,m_,points)
        error1 = error0
        error0 = c_
        print(c_)
        c.append(c_)
    return [b,m,c]

def run():
    points = genfromtxt('income.csv',delimiter=',')
    #hiperparemeter
    learning_rate = 0.001
    #y = mx + b
    initial_b = [0]
    initial_m = [0]
    n_interations = 50000
    b,m,c = gradient_descent_runner(points, initial_b, initial_m, n_interations, learning_rate)
    print(b[-1])
    print(m[-1])
    print(compute_error_for_given_point(b[-1],m[-1],points))
    plt.plot(c,range(len(b)))
    plt.xlabel("iterations")
    plt.ylabel("error")
    plt.show()

    function = points[:,0]*m[-1] + b[-1]
    plt.plot(points[:,0],function,color='black')
    plt.scatter(points[:,0],points[:,1],color='red')
    plt.show()



if __name__ == '__main__':
    run()

#3. O RSS dimiuiu. Porque a cada interacao ele diminui do valor do theta

#4. learning_rate = 0.001   n_interations = 50000

#6. 0.000001