from matplotlib.pyplot import *

from numpy import *

d = zeros((16, 40), float)

mapping = [[0], [1], [2], [3], [4], [5], [6], [7,8], [8, 9], [9,10], [10, 11, 12], [11, 12, 13], [13, 14, 15, 16], [15, 16, 17, 18], [17, 18, 19, 20, 21], [20, 21, 22, 23, 24, 25]]
print(len(mapping))
for i, l in enumerate(mapping):
  v = sqrt(1./len(l))
  for j in l:
    d[i,j] = v

imshow(d)
show()

