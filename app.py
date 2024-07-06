import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
url = 'house.csv'
df = pd.read_csv(url)

df_float = df.select_dtypes(include=['float64']).copy()
df_float.info()

df_float.describe()

df_float['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=False)
lotFrontage = df_float['LotFrontage']
salePrice = df['SalePrice']
x_train=np.array(lotFrontage)
y_train=np.array(salePrice)

def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(x, y, w, b): 
    
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w , b)     

     
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        if i<10000:      
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
 
    return w, b, J_history, p_history 

w_init = 0
b_init = 0
iterations = 223
tmp_alpha = 1e-4
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
print(f"65 lot frontage {w_final*65 + b_final:0.1f} Thousand dollars")