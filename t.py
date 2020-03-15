from numpy import *
from math import log2, log
from scipy.optimize import minimize
from matplotlib.pyplot import *

def f(i, base, offset1, offset2, mul):
  return ((log(i+offset1, base)-offset2)*mul+15)

def target_old(i):
  if i >= 1 and i <= 8:
    return i
  elif i >= 9 and i <= 16:
    return round((i-8)/2+8)
  elif i>= 17 and i <= 24:
    return round((i-16)/4+12)
  else:
    return round((i-24)/8 + 14)


target = dict((i+1, t) for i, t in enumerate([1,2,3,4,5,6,7,8,9,9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15])) 

def slot(x):
  t = 0
  for i, t in target.items():
    t += (f(i,x[0], x[1], x[2], x[3]) - t) ** 2 / (i**4)
  return t



x0 = array([1.2, 14., 14.8, 3.])

res = minimize(slot, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

print(res)

ps = []
for i in range(1,32):
  ps.append((f(i, res.x[0], res.x[1], res.x[2], res.x[3]), target[i]))


plot(ps)
show()
