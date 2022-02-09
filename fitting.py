import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
#https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
#https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
"""修改處"""
def func(x, a, b,c):
    return a*x**2+b*x+c
    
def readfile(n):#若n==1，跳過第一列
    x = []
    y = []
    with open('example.csv', newline='') as csvfile:    
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for row in rows:
            print(row)
            if n==1:
                n=0
                continue
            else:
                x.append(float(row[0]))
                y.append(float(row[1]))
    return x,y
"""修改處"""                
xdata,ydata = readfile(1)#若n==1，跳過第一列
x = np.array(xdata)
y = np.array(ydata)

# x = np.arange(1, 16, 1)
# num = [4.00, 5.20, 5.900, 6.80, 7.34,
       # 8.57, 9.86, 10.12, 12.56, 14.32,
       # 15.42, 16.50, 18.92, 19.58, 20.00]
# y = np.array(num)


popt, pcov = curve_fit(func, x, y)

yvals = func(x,*popt) #擬合y值
residuals = y- func(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y-np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)

"""修改處"""
print('係數a:', popt[0])
print('係數b:', popt[1])
print('係數c:', popt[2])
print('y = %0.3Ex^2+%0.3Ex+%0.3E'%(popt[0],popt[1],popt[2]))
print("R^2 =",r_squared)

#繪圖
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')

"""修改處"""
plt.title('curve_fit')
plt.xlabel('x')
plt.ylabel('y')
plt.text(0,0.05, '$y = %0.3Ex^2+%0.3Ex+%0.3E$'%(popt[0],popt[1],popt[2]),fontsize = 12,ha='left')# 設定文字
plt.text(0,-0.05, '$R^2 = %E$'%r_squared,fontsize = 12,ha='left')

plt.legend(loc=4) #指定legend的位置右下角
plt.show()
