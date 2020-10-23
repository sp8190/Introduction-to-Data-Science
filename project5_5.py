import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
import time


def load_dbscore_data():
    conn = pymysql.connect(host='localhost', user='root', password='tmd123', db='university')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    sql = "select * from db_score"
    curs.execute(sql)
    
    data  = curs.fetchall()
    
    curs.close()
    conn.close()
    
    #X = [ (t['attendance'], t['homework'], t['midterm'] ) for t in data ]
    X = [ ( t['attendance'],t['homework'],t['midterm'] ) for t in data ]
    X = np.array(X)
    
    X1 = [ ( t['attendance'] ) for t in data ]
    X1 = np.array(X1)
    X2 = [ ( t['homework']) for t in data ]
    X2 = np.array(X2)
    X3 = [ ( t['midterm'] ) for t in data ]
    X3 = np.array(X3)

    y = [ (t['score']) for t in data]
    y = np.array(y)

    return X, y, X1, X2, X3

X, y, X1, X2, X3 = load_dbscore_data()

import statsmodels.api as sm
X_const = sm.add_constant(X)

model = sm.OLS(y, X_const) # Ordinary Least Squares 결정론적 회귀 방법
ls = model.fit()

#print(ls.summary())


ls_c = ls.params[0]
ls_m1 = ls.params[1]
ls_m2 = ls.params[2]
ls_m3 = ls.params[3]

# c = 24.3479
# m = 1.6848

y_pred = ls_m1*X1 + ls_m2*X2 + ls_m3*X3 + ls_c
#plt.scatter(X, y) 
#plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')
#plt.show()

def gradient_descent_vectorized(X1, X2, X3, y):
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001
    
    m1 = 0.0
    m2 = 0.0
    m3 = 0.0
    c = 0.0
    
    n = len(y)
    
    c_grad = 0.0
    m1_grad = 0.0
    m2_grad = 0.0
    m3_grad = 0.0

    for epoch in range(epochs):    
    
        y_pred = m1 * X1 + m2 * X2 + m3 * X3 + c
        m1_grad = (2*(y_pred - y)*X1).sum()/n
        m2_grad = (2*(y_pred - y)*X2).sum()/n
        m3_grad = (2*(y_pred - y)*X3).sum()/n
        c_grad = (2 * (y_pred - y)).sum()/n
        
        m1 = m1 - learning_rate * m1_grad
        m2 = m2 - learning_rate * m2_grad
        m3 = m3 - learning_rate * m3_grad
        c = c - learning_rate * c_grad        

        if ( epoch % 1000 == 0):
            print("epoch %d: m1_grad=%f, m2_grad=%f, m3_grad=%f, c_grad=%f, m=%f, c=%f" \
                %(epoch, m1_grad, m2_grad, m3_grad, c_grad, m, c) )
    
        if ( abs(m1_grad) < min_grad and abs(m2_grad) < min_grad and\
             abs(m3_grad) < min_grad and abs(c_grad) < min_grad ):
            break

    return m1, m2, m3, c


start_time = time.time()
m1, m2, m3, c = gradient_descent_vectorized(X1, X2, X3, y)
end_time = time.time()

print("%f seconds" %(end_time - start_time))

print("\n\nFinal:")
print("gdv_m=%f, gdv_c=%f" %(m, c) )
print("ls_m=%f, ls_c=%f" %(ls_m, ls_c) )
