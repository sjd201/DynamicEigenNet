from SPVec import SPVec
from SPMemory import SPMemory
from scipy.sparse import *
import numpy as np
import history
from matplotlib import pyplot as plt
from time import time
from sys import stdout
from math import sqrt
import pyttsx3
import log
from random import randint

from speechinput import recognize_speech_from_mic

speechengine = pyttsx3.init()

corpus = []
lambdaO = 1.
lambdaS = 1.
lambdaSP = 10.

def resetMemory():
  global O, S, P, SP

  O = SPMemory(SPVec.maxSize*SPVec.NumberOfSlots, lambdaO)
  S = SPMemory(SPVec.maxSize, lambdaS)
  SP = [SPMemory(SPVec.maxSize*3, lambdaSP) for i in range(SPVec.NumberOfSlots)]

def encodeCorpus():
  global corpus, rawcorpus

  while rawcorpus != []:
    corpus.append(rawcorpus.pop(0))
    probe = " ".join(corpus[-SPVec.NumberOfSlots:])
    log.write("Encode " + probe)
    encode(probe)

def encode(probe):
  global o, s, sp
  slength = sqrt(SPVec.LengthOfMiller)
  zeroSP = SPVec("", Order=False).cat(SPVec("", Order=False).cat(SPVec("", Order=False))) # SP trace when encoding a blank
  toks = probe.split()
  toks += ["_"] * max(0, SPVec.NumberOfSlots - len(toks))
  probe = " ".join(toks)
  i = SPVec(probe)
  log.write("Add to Order memory " + str(i))
  O.addTrace(i) 
  o = O.getEcho(i, verbose=True)
  log.write("Order echo " + str(o))
  O.addTraceActivationsToHistory(toks[-1])
  si = SPVec(probe, Order=False).normalize()
  log.write("Add to Syntagmatic memory " + str(si))
  S.addTrace(si)
  s = S.getEcho(si).normalize()
  S.addTraceActivationsToHistory(toks[-1])
  sp = []
  for k in range(SPVec.NumberOfSlots):
    if toks[k] != "_":  # encode blanks as 0s
      ok = o.bank(k).normalize()
      sp.append(si.cat(ok.cat(i.bank(k))))
      log.write("Add to SP[%d] memory " % k + str(sp[k]))
      SP[k].addTrace(sp[k])
    else:
      sp.append(zeroSP)
      SP[k].addTrace(zeroSP)
    
def retrieveEcho(probe = "# who is bellamira loved by ? # _ _ _"):
  global o, s, sp, spechos
  slength = sqrt(SPVec.LengthOfMiller)
  res = SPVec("")
  i = SPVec(probe)
  log.write("getEcho O: " + probe)
  o = O.getEcho(i) 
  si = SPVec(probe, Order=False)
  log.write("getEcho S" + probe)
  s = S.getEcho(si).normalize() #+ si/slength
  sp = []
  toks = probe.split()
  spechos = []
  for j in range(SPVec.NumberOfSlots):
    oj = o.bank(j).normalize()
    sp.append(s.cat(oj.cat(SPVec("", Order = False))))
    log.write("getEcho SP slot %d" % j + " " + probe)
    specho = SPVec(SP[j].getEcho(sp[j]))
    spechos.append(specho)
    SP[j].addTraceActivationsToHistory(toks[-1], toks[j])
    res.setBank(j, spechos[j].bank(2))
  return res

def getWords(echo, showActivations = False):
  res = []
  for j in range(SPVec.NumberOfSlots):
    res += [echo.bank(j).maxItems(showActivations=showActivations)]
  return res

def retrieve(probe = "# who is bellamira loved by ? # _ _ _", showActivations = False):
  echo = retrieveEcho(probe)
  return getWords(echo, showActivations)

def load(filename):
  global rawcorpus

  resetMemory()
  rawcorpus = open(filename, "r").read()
  rawcorpus = rawcorpus.lower().split()
  log.write("corpus size = " + str(len(rawcorpus)) + " vocab size = " + str(len(set(rawcorpus))))
  encodeCorpus()

def showCorpus():
  res = " ".join(corpus)
  print("\n#".join(res.split("#")))

load("tiny.cor")

def spinput(voice=False, PROMPT_LIMIT=5):
  if voice:
    for j in range(PROMPT_LIMIT):
      print("Wating for input")
      guess = recognize_speech_from_mic()
      if guess["transcription"]:
        break
      if not guess["success"]:
        break
      r = randint(1,3)
      if r == 1:
        print("Output: I didn't catch that")
        speechengine.say("I didn't catch that")
      elif r == 2:
        print("Output: Sorry")
        speechengine.say("Sorry")
      else:
        print("Output: What did you say")
        speechengine.say("What did you say")
      speechengine.runAndWait()
    inp = guess["transcription"]
  else:
    inp = input("> ")
  return inp

def spoutput(speechoutput, voice=True):
  if voice:
    if speechoutput != "":
      speechengine.say(speechoutput)
      speechengine.runAndWait()
  else:
      print(s)

def conversation():
  global rawcorpus

  spoutput("good morning sir ")
  inp = ""
  while inp.lower() != "goodbye jamie":
    inp = spinput()
    print("Input: ", inp)
    if inp.lower() != "goodbye jamie":
        inputtoks = inp.split()
        if inputtoks[0] == "jamie":
          if inputtoks[1] == "show":
            if inputtoks[2] == "the":
              if inputtoks[3] == "order":
                if inputtoks[4] == "traces" and inputtoks[5] == "from" and inputtoks[6] == "the" and inputtoks[8] == "window":
                  print(O.getTraceActivationsFromHistory(inputtoks[7]))
              elif inputtoks[3] == "syntagmatic":
                if inputtoks[4] == "traces" and inputtoks[5] == "from" and inputtoks[6] == "the" and inputtoks[8] == "window":
                  print(S.getTraceActivationsFromHistory(inputtoks[7]))
              elif inputtoks[3] == "sp":
                if inputtoks[4] == "traces" and inputtoks[5] == "from" and inputtoks[6] == "the" and inputtoks[8] == "window":
                  for i in range(SPVec.NumberOfSlots):
                    print(SP[i].getTraceActivationsFromHistory(inputtoks[7], i))
            elif inputtoks[2] == "corpus":
              showCorpus()
              
        else:
          inp = "# " + inp + " #"
          rawcorpus = inp.split()
          encodeCorpus()
          suffix = ["_"] * 3
          probe = " ".join(corpus[-SPVec.NumberOfSlots:] + suffix)
          log.write("Probe: " + probe)
          output = retrieve(probe)[-len(suffix):]
          log.write("Output: "+ " ".join(output))
          
          while "#" not in output and len(suffix) < 8:
            suffix += ["_"]
            probe = " ".join(corpus[-SPVec.NumberOfSlots:] + suffix)
            log.write( "Probe: "+ probe)
            output = retrieve(probe)[-len(suffix):]
            log.write( "Output: "+ " ".join(output))
          i = -len(suffix)+1
          while i < 0 and output[i] != "#":
            i += 1
          rawcorpus = output[-len(suffix):i]
          speechoutput = " ".join(rawcorpus)
          print("Output: ", speechoutput)
          rawcorpus += "# _ _ _ _ _ _ _ _ _ _ _ _ _ _".split()
          encodeCorpus()
          
          spoutput(speechoutput)
  spoutput("goodbye sir ")

conversation()
