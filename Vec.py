import collections
from scipy.sparse import csr_matrix, csc_matrix

from numpy import argsort, array, sqrt, int64, float64, ndarray
from traceback import print_stack
import itertools
import log
import shelve
import random

stoplist = "does im hi did who is love loved by ? # loves the from buy sold to . ok cool".split()

class Vec(csr_matrix):
    store = dict()
    vocablist = []
    maxSize = 700
    MaxItemsToShow = 6
    ShowValues = True
    OutputThreshold = 0.4 # the threshold the value of the maximum activation in a vector must reach in order for it to be output
    #mapping = [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 10], [9, 10, 11, 12], [12, 13, 14], [13, 14, 15], [15, 16], [16, 17], [17, 18], [19], [20], [21], [22], [23], [24], [25]]
    #NumberOfToks = 26
    OrderMapping = [[0, 1, 2], [1, 2], [2], [3], [4], [5], [6], [7], [8], [9]]
    NumberOfToks = 10
    NumberOfSlots = 10
    LengthOfMiller = 12
    Length = maxSize * (NumberOfSlots+1)

    def __init__(self, s, shape=None, dtype=None, copy=None):
        if type(s) == str:
            csr_matrix.__init__(self, (1, Vec.Length), shape=shape, dtype=float64, copy=copy)
            toks = s.split()
            self.encode(toks)
        elif type(s) == list:
            csr_matrix.__init__(self, (1, Vec.Length), shape=shape, dtype=float64, copy=copy)
            self.encode(s)
        elif type(s) == csr_matrix or type(s) == ndarray:
          csr_matrix.__init__(self, s, shape=shape, dtype=dtype, copy=copy)
        elif type(s) == int:
          csr_matrix.__init__(self, (1,s), shape=shape, dtype=dtype, copy=copy)
        elif type(s) == tuple:
          csr_matrix.__init__(self, s, shape=shape, dtype=dtype, copy=copy)
        else:
          raise Exception("Unknown argument type for Vec: " + str(type(s)))

    def encode(self, toks):
      if len(toks) < Vec.NumberOfToks:
        localtoks = ["_"] * (Vec.NumberOfToks - len(toks)) + toks
      else:
        localtoks = toks[(-Vec.NumberOfToks):]
      for slot, l in enumerate(Vec.OrderMapping):
        for pos in l:
          v = sqrt(1./len(l))
          if localtoks[pos] != "_":
            tok_index = self.__getindexorname__(localtoks[pos])
            self[0, slot*Vec.maxSize+tok_index] = v
            if localtoks[pos] == "one":
                self[0, slot*Vec.maxSize+tok_index] = 1.0

      # add the Miller bank

      #for i in range(len(toks)-1, len(toks)-Vec.LengthOfMiller - 1, -1):
      #  if i >= 0:
      #    if toks[i] != "_" and toks[i] not in stoplist:
      #      tok_index = self.__getindexorname__(toks[i])
      #      self[0, Vec.maxSize*Vec.NumberOfSlots+tok_index] = 1.

      # add random bank

      #for i in xrange(100):
      #    self[0, Vec.maxSize*Vec.NumberOfSlots+Vec.maxSize-100+i] = 0.0 #random.randint(-1, 1) / sqrt(10.)

    def save(directoryname):
        s = shelve.open(directoryname+"/Vec.par")
        s["store"] = Vec.store 
        s["vocablist"] = Vec.vocablist 
        s.close()

    def load(directoryname):
        s = shelve.open(directoryname+"/Vec.par")
        Vec.store = s["store"] 
        Vec.vocablist = s["vocablist"]
        s.close()

    def __contains__(self, key):
        return(key in Vec.store)

    def __getindexorname__(self, x):
        if isinstance(x, int64):
          if x >= 0 and x < len(Vec.vocablist):
            return(Vec.vocablist[x])
          else:
            raise Exception("Index must be >= 0 and < len(vocab). Index = %d len(vocab) = %d." % (x, len(Vec.vocablist)))
        else:
          newkey = self.__keytransform__(x)
          if newkey not in Vec.store:
            if len(Vec.store) < Vec.maxSize:
              Vec.store[newkey] = len(Vec.store)
              Vec.vocablist.append(newkey)
            else:
              return Vec.store["__default__"]
          return Vec.store[newkey]
          
    def __iter__(self):
        return iter(Vec.store)

    def __len__(self):
        return Vec.maxSize

    def __keytransform__(self, key):
        if isinstance(key, str):
          return(key.lower())
        elif isinstance(key, tuple):
          if all(isinstance(v, basestring) for v in key):
            return(tuple([v.lower() for v in key]))
        return key

    def cat(self, vec):
        res = csr_matrix((1, self.shape[1]+vec.shape[1]))
        res[0, 0:self.shape[1]] = self
        res[0, self.shape[1]: (self.shape[1]+vec.shape[1])] = vec
        return Vec(res)

    def strList(self, l):
        if Vec.ShowValues:
          if len(l) > Vec.MaxItemsToShow:
            res = " ".join("%s %1.2f" % (w, v) for v, w in l[0:Vec.MaxItemsToShow] if v > 0.001) + " ..."
          else:
            res = " ".join("%s %1.2f" % (w, v) for v, w in l if v > 0.001) 
        else:
          if len(l) > Vec.MaxItemsToShow:
            res = " ".join(w for v, w in l[0:Vec.MaxItemsToShow] if v > 0.001) + " ..."
          else:
            res = " ".join(w for v, w in l if v > 0.001) 
        return res

    def maxItems(self):
      res = ""
      for b in range(int(self.shape[1]/self.maxSize)):
        thebank = self.bank(b)
        argm = thebank.argmax()
        if thebank[0, argm] < Vec.OutputThreshold:
          res += "_"
        else:
          res += str(self.__getindexorname__(argm))
      return res

    def bank(self, b):
      return Vec(self[0, (self.maxSize * b):(self.maxSize*(b+1))])

    def setBank(self, b, vec):
      self[0, (self.maxSize * b):(self.maxSize*(b+1))] = vec

    def addToBank(self, b, vec):
      self[0, (self.maxSize * b):(self.maxSize*(b+1))] += vec

    def __str__(self):
      res = [""] * int(self.shape[1]/self.maxSize)
      l = [[] for _ in range(int(self.shape[1]/self.maxSize))]
      cx = self.tocoo()    
      
      for i,j,v in zip(cx.row, cx.col, cx.data):
        bank = int(j / Vec.maxSize)
        tok_index = j % Vec.maxSize
        try:
          tok = self.__getindexorname__(tok_index)
        except:
          #tok = "S" + str(tok_index)
          tok = ""
        l[bank].append((v, tok))
      for i, s in enumerate(l):
        s.sort(reverse=True)
        res[i] = self.strList(s)
      return "| "+" | ".join(res) + " |"

    def __add__(self, vec):
      return Vec(csr_matrix(self) + csr_matrix(vec))
  
    def __mul__(self, v):
      if type(v) == csc_matrix:
        return Vec(csr_matrix(self) * csr_matrix(v))
      elif type(v) == float64 or type(v) == float: # scalar
        return Vec(csr_matrix(self) * v)
      else:
        raise Exception("Vec.__mul__ received a {} argument.".format(str(type(v))))

    def max(self):
      return self.max()
  
    def argmax(self):
      c = csr_matrix(self)
      return c.argmax()
  
    def length(self):
      if self.nnz == 0:
        return 0.0
      else:
        return sqrt((self * self.transpose())[0,0])

    def normalize(self):
      l = self.length()
      return self * (1. / (l+0.000001))

if __name__ == "__main__":
  orders = "hi there this is a test of the creation of order vectors from a string a tilt representation is used which becomes less precise as the past recedes"
  print orders
  vec = Vec(orders)
  print vec
  vec3 = vec.normalize()
  print vec3
