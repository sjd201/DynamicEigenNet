from Vec import Vec
from scipy.sparse import lil_matrix
import numpy as np
from math import exp
from copy import deepcopy
import log

class SPMemory():

  MaxNumberOfTraces = 1000

  def __init__(self, vecLength, mylambda = 1.0):
    self.M = lil_matrix((SPMemory.MaxNumberOfTraces, vecLength))
    self.NumberOfTraces = 0
    self.TraceActivationHistory = []
    self.TraceActivations = None
    self.mylambda = mylambda

  def addTraceActivationsToHistory(self, endofframeword, word=None):
      self.TraceActivationHistory.append((endofframeword, word, deepcopy(self.TraceActivations)))

  def getTraceActivationsFromHistory(self, endofframeword, indexinframe=None):

    for index in range(len(self.TraceActivationHistory)-1, 0, -1):
        if self.TraceActivationHistory[index][0] == endofframeword:
            break

    result = "Segment: " 
    result += " " + endofframeword + " "
    result += "%d " % index
    result += " %d " % indexinframe
    result += " %d " % len(self.TraceActivationHistory)
    result += "\n"
    for l in range(len(self.TraceActivationHistory)-100, len(self.TraceActivationHistory)):
      result += self.TraceActivationHistory[l][0] + " "
    result += "\n"
    start = max(0, index-Vec.NumberOfSlots)
    result +=  " ".join(t[0] for t in self.TraceActivationHistory[start:(index+1)]) + "\n"
 
    if indexinframe != None:
      word = self.TraceActivationHistory[index - Vec.NumberOfSlots + indexinframe + 1][0]
      result += word + "\n"
    print(self.TraceActivationHistory[index])
    if self.TraceActivationHistory[index][2].nnz > 0:
      res = self.TraceActivationHistory[index][2]
      reorder = np.argsort(res.data)[::-1]
      res.data = res.data[reorder]
      res.indices = res.indices[reorder]
      for i in range(len(res.indices)):
        realind = res.indices[i]
        if res.data[i] > 0.01 and i < 10:
          result +=  "%1.3f" % res.data[i] + " "+ str(Vec(self.M[realind])) + "\n"
    else:
      result += "TraceActivations has %d non zeros" % self.TraceActivationHistory[index][2].nnz
    return result

  def strTraceActivations(self):
    result = ""
    res = self.TraceActivations
    if res.nnz > 0:
      reorder = np.argsort(res.data)[::-1]
      res.data = res.data[reorder]
      res.indices = res.indices[reorder]
      for i in range(len(res.indices)):
        realind = res.indices[i]
        if res.data[i] > 0.01 and i < 10:
          result +=  "%1.3f" % res.data[i] + " "+ str(Vec(self.M[realind])) + "\n"
    else:
      result += "TraceActivations has %d non zeros" % res.nnz
    return result
    
  def addTrace(self, vec):
    if self.NumberOfTraces < SPMemory.MaxNumberOfTraces-1:
      self.M[self.NumberOfTraces] = vec
      self.NumberOfTraces += 1
    else:
      raise Exception("Memory out of traces.")
 
  def topn(self, N = 20):
    n = min(len(self.TraceActivations.data), N)
    ind = np.argpartition(self.TraceActivations.data, -n)[-n:]
    resinds = self.TraceActivations.indices[ind]
    resdata = self.TraceActivations.data[ind]
    self.TraceActivations.indices[:] = 0
    self.TraceActivations.data[:] = 0.0
    self.TraceActivations.indices[0:n] = resinds
    self.TraceActivations.data[0:n] = resdata
    self.TraceActivations.eliminate_zeros()

  def retrieveFromMemory(self, probe, verbose=False):
    '''
    Return a sparse matrix with the retrieval strengths of each of the traces.
    '''

    self.TraceActivations = self.M * probe.transpose()
    self.TraceActivations = self.TraceActivations.transpose().tocsr()
    self.topn()
    self.TraceActivations = self.TraceActivations * self.mylambda
    self.TraceActivations = self.TraceActivations.expm1()
    self.TraceActivations = self.TraceActivations/(self.TraceActivations.sum()+0.0000001)
    log.write(self.strTraceActivations())

  def getEcho(self, vec, verbose=False):
    self.retrieveFromMemory(vec, verbose=verbose)
    echo = self.TraceActivations * self.M
    return Vec(echo.tolil())

  def showActiveTraces(self, corpus = None, showTraceNumber = False):
      if self.TraceActivations.nnz > 0:
        res = self.TraceActivations
        reorder = np.argsort(res.data)[::-1]
        res.data = res.data[reorder]
        res.indices = res.indices[reorder]
        for i in range(len(res.indices)):
          realind = res.indices[i]
          if res.data[i] > 0.1 or i < 4:
            if showTraceNumber:
              print("%4d " % realind, Vec(self.M[realind]))
            if corpus:
              print("%1.7f" % res.data[i], " ".join(corpus[realind:(realind+Vec.NumberOfSlots)]))
            else:
              print("%1.7f" % res.data[i], Vec(self.M[realind]))
      else:
        print("TraceActivations has %d non zeros" % self.TraceActivations.nnz)

  def showTraces(self, corpus=None):
      for i in range(self.NumberOfTraces):
        if corpus:
          print(" ".join(corpus[realind:(realind+Vec.NumberOfSlots)]))
        else:
          print(Vec(self.M[i]))

  def listTraces(self, corpus):
    for i in range(self.NumberOfTraces):
      print("%4d %s" % (i, " ".join(corpus[i:(i+Vec.NumberOfSlots)])))
      print("%4d" % i, Vec(self.M[i]))

  def __str__(self):
    res = "Memory shape = %d %d NumberOfTraces = %d" % (self.M[0], self.M[1], self.NumberOfTraces)
    return(res)


