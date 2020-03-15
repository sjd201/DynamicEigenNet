from Vec import Vec
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.linalg import eigs
import numpy as np
from numpy.linalg import norm
import log
import shelve

class Memory():

  def __init__(self):
      self.M = csr_matrix((Vec.Length, Vec.Length))

  def eig(self):
      vtp1 = np.ones((Vec.Length, 1), float)
      vt = np.zeros((Vec.Length, 1), float)
      while norm(vtp1 - vt) > 1.e-15:
        vt = vtp1
        vtp1 = self.M.dot(vt)
        f = norm(vtp1)
        vtp1 = vtp1 / f
      return f, vtp1

  def eigenvalue(self):
      val, vec = self.eig()
      return val

  def stp(self, v): # can we make this faster by keeping the M and the stp part separate during eigenvalue computation rather than adding and subtracting??
      out = v.transpose().dot(v)
      self.M += out
      val, vec = self.eig()
      self.M -= out
      vec = vec.transpose()
      vec = csr_matrix(vec)
      return Vec(vec)

  def add(self, v):
      out = v.transpose().dot(v)
      self.M += out

  def __idiv__(self, val):
      self.M /= val
      return self

  def __str__(self):
    res = "Memory shape = %d %d" % (self.M.shape[0], self.M.shape[1])
    return(res)

  def save(self, directoryname):
    save_npz(directoryname + "/Memory.npz", self.M)

  def load(self, directoryname):
    self.M = load_npz(directoryname + "/Memory.npz")

if __name__ == "__main__":
  M = Memory()
  vec = Vec("oneandone one one zero").normalize()
  print vec
  M.add(vec)
  
  vec = Vec("oneandzero one zero one").normalize()
  print vec
  M.add(vec)
  vec = Vec("zeroandone zero one one").normalize()
  print vec
  M.add(vec)
  vec = Vec("zeroandzero zero zero zero").normalize()
  print vec
  M.add(vec)
  val = M.eigenvalue()
  M /= val


  vals, vecs = eigs(M.M, 4)
  print vals
  for i in xrange(4):
    print Vec(csr_matrix(vecs[:,i]))
    print

  # test

  testpatterns = ["oneandone one one _", "oneandzero one zero _", "zeroandone zero one _", "zeroandzero zero zero _"]
  for testpattern in testpatterns:
      vec = Vec(testpattern)
      vec = vec.normalize()
      print testpattern
      print M.stp(vec)
