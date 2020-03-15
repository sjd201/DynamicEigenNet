import collections
from scipy.sparse import *

from numpy import argsort, array, sqrt, int64
from traceback import print_stack
import itertools

stoplist = "im hi did who is loved by ? # loves the from buy sold to . ok cool".split()

class SPVec(lil_matrix):
    store = dict()
    vocablist = []
    maxSize = 125
    NumberOfSlots = 8
    LengthOfMiller = 8
    MaxItemsToShow = 2
    ShowValues = True#False
    OutputThreshold = 0.3 # the threshold the value of the maximum activation in a vector must reach in orderfor it to be output

    def __init__(self, s):
        if type(s) == str:
          lil_matrix.__init__(self, (1, SPVec.maxSize+SPVec.maxSize*SPVec.NumberOfSlots), dtype=float)
      
        elif type(s) == lil_matrix:
          lil_matrix.__init__(self, s)
        elif type(s) == SPVec:
          lil_matrix.__init__(self, s)
        elif type(s) == csr_matrix:
          lil_matrix.__init__(self, s.tolil())
        elif type(s) == int:
          lil_matrix.__init__(self, (1,s))
        else:
          raise Exception("Unknown argument type for SPVec: " + str(type(s)))

    def __contains__(self, key):
        return(key in SPVec.store)

    def __getindexorname__(self, x):
        if isinstance(x, int64):
          if x >= 0 and x < len(SPVec.vocablist):
            return(SPVec.vocablist[x])
          else:
            raise Exception("Index must be >= 0 and < len(vocab). Index = %d len(vocab) = %d." % (x, len(SPVec.vocablist)))
        else:
          newkey = self.__keytransform__(x)
          if newkey not in SPVec.store:
            if len(SPVec.store) < SPVec.maxSize:
              #if self.itemvocab:
              #  print self.itemvocab[newkey[0]], self.itemvocab[newkey[1]]
              #  print_stack()
              SPVec.store[newkey] = len(SPVec.store)
              SPVec.vocablist.append(newkey)
            else:
              return SPVec.store["__default__"]
          return SPVec.store[newkey]
          
    def __iter__(self):
        return iter(SPVec.store)

    def __len__(self):
        return SPVec.maxSize

    def __keytransform__(self, key):
        if isinstance(key, str):
          return(key.lower())
        elif isinstance(key, tuple):
          if all(isinstance(v, basestring) for v in key):
            return(tuple([v.lower() for v in key]))
        return key

    def encode(self, s):
      toks = s.split()
      for i in range(len(toks)-1, len(toks)-SPVec.NumberOfSlots - 1, -1):
        if i >= 0:
          if toks[i] != "_":
            tok_index = self.__getindexorname__(toks[i])
            slot = i - len(toks)+SPVec.NumberOfSlots
            self[0, slot*SPVec.maxSize+tok_index] = 1.0
      for i in range(len(toks)-1, len(toks)-SPVec.LengthOfMiller - 1, -1):
        if i >= 0:
          if toks[i] != "_" and toks[i] not in stoplist:
            tok_index = self.__getindexorname__(toks[i])
            self[0, SPVec.maxSize*SPVec.NumberOfSlots+tok_index] = 1.

    def cat(self, vec):
        res = lil_matrix((1, self.shape[1]+vec.shape[1]))
        res[0, 0:self.shape[1]] = self
        res[0, self.shape[1]: (self.shape[1]+vec.shape[1])] = vec
        return SPVec(res)

    def strList(self, l):
        if SPVec.ShowValues:
          if len(l) > SPVec.MaxItemsToShow:
            res = " ".join("%s %1.2f" % (w, v) for v, w in l[0:SPVec.MaxItemsToShow]) + " ..."
          else:
            res = " ".join("%s %1.2f" % (w, v) for v, w in l) 
        else:
          if len(l) > SPVec.MaxItemsToShow:
            res = " ".join(w for v, w in l[0:SPVec.MaxItemsToShow]) + " ..."
          else:
            res = " ".join(w for v, w in l) 
        return res

    def maxItems(self):
      res = ""
      for b in range(int(self.shape[1]/self.maxSize)):
        thebank = self.bank(b)
        argm = thebank.argmax()
        if thebank[0, argm] < SPVec.OutputThreshold:
          res += "_"
        else:
          res += str(self.__getindexorname__(argm))
      return res

    def bank(self, b):
      return SPVec(self[0, (self.maxSize * b):(self.maxSize*(b+1))])

    def setBank(self, b, vec):
      self[0, (self.maxSize * b):(self.maxSize*(b+1))] = vec

    def addToBank(self, b, vec):
      self[0, (self.maxSize * b):(self.maxSize*(b+1))] += vec

    def __str__(self):
      res = [" "] * int(self.shape[1]/self.maxSize)
      cx = self.tocoo()    
      
      l = []
      lastbank = 0
      bank = 0
      for i,j,v in zip(cx.row, cx.col, cx.data):
        bank = j / SPVec.maxSize
        tok_index = j % SPVec.maxSize
        tok = self.__getindexorname__(tok_index)
        if lastbank < bank:
          l.sort(reverse=True)
          res[int(lastbank)] = self.strList(l)
          l = []
          lastbank = bank

        l += [(v,tok)]
      l.sort(reverse=True)
      res[int(bank)] = self.strList(l)
      return "| "+" | ".join(res) + " |"

    def __add__(self, vec):
      return SPVec(lil_matrix(self.tocsr() + vec))
  
    def __mul__(self, v):
      if type(v) == lil_matrix:
        return self.tocsr() * v
      else:
        return SPVec(lil_matrix(v * self.tocsr()))

    def max(self):
      return self.tocsr().max()
  
    def argmax(self):
      return self.tocsr().argmax()
  
    def length(self):
      return sqrt(self * self.transpose())[0,0]

    def normalize(self):
      l = self.length()
      return SPVec(self * (1. / (l+0.000001)))

if __name__ == "__main__":
  s = "#"
  print(s)
  vec = SPVec(s)
  print(vec)
  s2 = "who"
  vec2 = SPVec(s2)
  print(vec2)
  vec3 = vec.normalize()
  vec4 = vec2.normalize()
  vec5 = (vec3 + vec4) * (1./2)
  print(vec5)
  print(vec5.length())
  print SPVec("hello there . my name is simon")
