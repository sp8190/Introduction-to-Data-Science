import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
from matplotlib import animation


def load_dbscore_data():
    conn = pymysql.connect(host='localhost', user='root', password='tmd123', db='university')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    sql = "select * from db_score"
    curs.execute(sql)
    
    data  = curs.fetchall()
    
    curs.close()
    conn.close()
    
    #X = [ (t['attendance'], t['homework'], t['midterm'] ) for t in data ]
    X = [ ( t['midterm'] ) for t in data ]
    X = np.array(X)
    
    y = [ (t['score']) for t in data]
    y = np.array(y)

    return X, y

def gradient_descent_vectorized(X, y):
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001
    
    m = 0.0
    c = 0.0
    
    n = len(y)
    
    c_grad = 0.0
    m_grad = 0.0
    m_list = []
    c_list = []

    for epoch in range(epochs):    
    
        y_pred = m * X + c
        m_grad = (2*(y_pred - y)*X).sum()/n
        c_grad = (2 * (y_pred - y)).sum()/n
        
        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad

        if ( epoch % 500 == 0):
            m_list.append(m)
            c_list.append(c)
    
        if ( abs(m_grad) < min_grad and abs(c_grad) < min_grad ):
            break

    return m_list, c_list


X, y = load_dbscore_data()


###########################
# plt.scatter(X, y) 
# plt.show()

import statsmodels.api as sm
X_const = sm.add_constant(X)

model = sm.OLS(y, X_const) # Ordinary Least Squares 결정론적 회귀 방법
ls = model.fit()

# print(ls.summary())

ls_c = ls.params[0]
ls_m = ls.params[1]

y_pred = ls_m*X + ls_c
# plt.scatter(X, y) 
# plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')
###########################


fig, ax = plt.subplots()
ax.grid(False)

line, = ax.plot([], [], lw=2)
m = []
c = []

m, c = gradient_descent_vectorized(X, y)



def init():
    plt.scatter(X, y,color='red')
    line.set_data([], [])
    return (line,)

def animate(t,m,c):
    print(t)
    #0과 2사이 1000개 점 찍기
    nx = np.linspace(0, 40, 10)
    ny = np.array(float(m[t]) * nx + float(c[t]))
    mc_str = "m={0},\nc={1}".format(m[t],c[t])
    plt.text(-3, 60,'■',color='white',fontsize=163)
    plt.text(-1, 88, mc_str,fontsize=8)
    
    line.set_data(nx,ny)
    return (line,)

ani = animation.FuncAnimation(fig, animate, init_func=init,\
    frames=108,fargs=(m,c),interval=30,blit=True)

ani.save('exAnimation.gif', writer='imagemagick', fps=30, dpi=100)

plt.show()