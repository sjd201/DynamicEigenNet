from Vec import OrderVec, SyntagmaticVec, SPVec, Vec
from Memory import OrderMemory, SyntagmaticMemory, SPMemory, Memory
from scipy.sparse import save_npz, load_npz
import numpy as np
import history
from matplotlib import pyplot as plt
from time import time
from sys import stdout, maxsize
from math import sqrt
import log
from random import randint
import warnings
warnings.filterwarnings("ignore")
from gtts import gTTS
import vlc
import os


from speechinput import recognize_speech_from_mic

#speechengine = pyttsx3.init()

corpus = []
OrderThreshold = 0.3 # threshold that length of retreived order information must reach before it is used as a retreival cue to SP memory

def resetMemory():
  global O, S, SP

  O = OrderMemory("order")
  S = SyntagmaticMemory("syntagmatic")
  SP = [SPMemory("SP{}".format(i)) for i in range(OrderVec.NumberOfSlots)]

def encodeCorpus():
  global corpus, rawcorpus

  while rawcorpus != []:
    word = rawcorpus.pop(0)
    corpus.append(word)
    #probe = " ".join(corpus[-OrderVec.NumberOfSlots:])
    #probe = corpus[-OrderVec.NumberOfSlots:]
    #log.write("")
    #log.write("Encode " + " ".join(corpus[-OrderVec.NumberOfToks:]))
    encode(corpus)

def encode(toks):
  global o, s, sp
  toks = toks[-OrderVec.NumberOfToks:]
  toks = ["_"] * max(0, OrderVec.NumberOfToks - len(toks)) + toks
  #probe = " ".join(toks)
  i = OrderVec(toks)
  #O.addTrace(i) 
  o = O.getEcho(i, verbose=True)
  si = SyntagmaticVec(toks[-SyntagmaticVec.LengthOfMiller:]).normalize()
  #S.addTrace(si)
  s = S.getEcho(si).normalize()
  O.addTrace(i) 
  S.addTrace(si)
  sp = []
  slottoks = toks[-OrderVec.NumberOfSlots:]
  for k in range(OrderVec.NumberOfSlots):
    #log.write("SP Slot %d:" % k)
    if slottoks[k] != "_":  # encode blanks as 0s
      ok = o.bank(k).normalize()
      sp.append(si.cat(ok.cat(i.bank(k))))
      SP[k].addTrace(sp[k])
    else:
      sp.append(SPVec.zero())
      SP[k].addTrace(SPVec.zero())
    
def retrieve(probe):
  global o, s, sp, spechos
  res = []
  #res = OrderVec([])
  i = OrderVec(probe)
  o = O.getEcho(i) 
  si = SyntagmaticVec(probe)
  s = S.getEcho(si)
  for j in range(OrderVec.NumberOfSlots):
    oj = o.bank(j)
    slotlength = oj.length()
    if slotlength > OrderThreshold:
      sp = s.cat(oj.cat(SyntagmaticVec("")))
      specho = SP[j].getEcho(sp)
      out = specho.bank(2)
      res.append(out.maxItems())
    else:
      res.append("_")
  return res

#def getWords(echo):
#  res = []
#  for j in range(OrderVec.NumberOfSlots):
#    res += [echo.bank(j).maxItems()]
#  return res

#def retrieve(probe):
#  echo = retrieveEcho(probe)
#  return getWords(echo)

def corpusExists(directoryname):
  return os.path.isfile(directoryname+"/corpus")

def corpusCompiled(directoryname):
  # assume directoryname/corpus exists
  if os.path.isfile(directoryname + "/order.npz"):
    return os.path.getmtime(directoryname+'/corpus') < os.path.getmtime(directoryname+'/order.npz')
  else:
    return False
  
def load(directoryname=""):
  global rawcorpus, currentdirectoryname

  if directoryname == "":
    directoryname = currentdirectoryname
  else:
    currentdirectoryname = directoryname

  if not corpusExists(directoryname):
    print(directoryname+"/corpus does not exist")
  else:

    resetMemory()
    if corpusCompiled(directoryname):
      Vec.load(directoryname)
      O.load(directoryname)
      S.load(directoryname)
      for i in range(OrderVec.NumberOfSlots):
        SP[i].load(directoryname)
    else:
      rawcorpus = open(directoryname+"/corpus", "r").read()
      rawcorpus = rawcorpus.lower().split()
      log.write("corpus size = " + str(len(rawcorpus)) + " vocab size = " + str(len(set(rawcorpus))))
      encodeCorpus()
      O.save(directoryname)
      S.save(directoryname)
      [SP[i].save(directoryname) for i in range(OrderVec.NumberOfSlots)]
      Vec.save(directoryname)
    print(directoryname + " loaded")

load("tiny")

def spinput(voice=True, PROMPT_LIMIT=5):
  if voice:
    for j in range(PROMPT_LIMIT):
      stdout.write('You: ')
      stdout.flush()
      guess = recognize_speech_from_mic()
      if guess["transcription"]:
        break
      if not guess["success"]:
        break
      r = randint(1,3)
      if r == 1:
        spoutput("I didn't catch that")
      elif r == 2:
        spoutput("Sorry")
      else:
        spoutput("What did you say")
    inp = guess["transcription"]
    inp = inp.replace("equals", "=")
    print(inp)
  else:
    inp = input("You: ")
  return inp

def spoutput(speechoutput, voice=True):
  print("Jamie: " + speechoutput)
  if voice:
    if speechoutput != "":
      #speechengine.say(speechoutput)
      #speechengine.runAndWait()
      myobj = gTTS(text=speechoutput, lang="en", slow=False)
      myobj.save("speechoutput.mp3")
      #os.system("mpg321 speechoutput.mp3")
      p = vlc.MediaPlayer("./speechoutput.mp3")
      p.play()
      # wait until the response has been played so that the speech 
      # recognition software doesn't start transcribing the response 
      # as the next input from the user
      Ended = 6
      current_state = p.get_state()
      while current_state != Ended:
        current_state = p.get_state()

def conversation():
  global rawcorpus

  spoutput("good morning sir ")
  inp = ""
  while inp.lower() != "goodbye jamie":
    pass

def processCommand(toks):
      if toks[1] == "help":
        log.write("jamie help")
        log.write("jamie show values")
        log.write("jamie reset/reload")
        log.write("jamie events to log")
        log.write("jamie [don't] show <memoytype> <datatype> during <phase> phase")
        log.write("memorytype = order|syntagmatic|sp")
        log.write("datatype = probes|echoes|activations")
        log.write("phase = my encoding|your encoding|your retrieval")
        log.write("e.g. jamie show order activations during your retrieval")
        log.write("e.g. jamie don't show syntagmatic probes during my encoding")
        log.write("e.g. jamie show sp probes")
      elif toks[1] == "load":
        if len(toks) > 2:
          load(toks[2])
      elif toks[1] == "show":
        if toks[2] == "values":
          Vec.ShowValues = True
        elif toks[2] == "events" and toks[3] == "to" and toks[4] == "log":
          log.write("Events to log:")
          for event in log.eventstolog:
            log.write(event)
        else:
          if len(toks) > 2:
            if log.isMemoryType(toks[2]):
              if len(toks) > 3:
                if log.isDataType(toks[3]):
                  if len(toks) > 6:
                    phase = toks[5] + " " + toks[6]
                    if log.isPhase(phase):
                      log.eventstolog.add(toks[2] + " " + toks[3] + " " + phase)
                      log.write("show " + toks[2] + " " + toks[3] + " " + phase)
                    else:
                      spoutput(phase + " is not a known phase")
                  else:
                    for phase in log.phasestolog:
                      log.eventstolog.add(toks[2] + " " + toks[3] + " " + phase)
                      log.write("show " + toks[2] + " " + toks[3] + " " + phase)
                else:
                  spoutput(toks[3] + " is not a known data type. try probes echoes or activations")
              else:
                spoutput("You didn't provide a data type, try probes echoes or activations")
            else:
              spoutput(toks[2] + " is not a known memory type. try order syntagmatic or sp")
          else:
            spoutput("You didn't provide a memory type, try order syntagmatic or sp")
      elif toks[1] == "don't" and toks[2] == "show":
        if toks[3] == "values":
          Vec.ShowValues = False
        else:
          if len(toks) > 3:
            if log.isMemoryType(toks[3]):
              if len(toks) > 4:
                if log.isDataType(toks[4]):
                  if len(toks) > 7:
                    phase = toks[6] + " " + toks[7]
                    if log.isPhase(phase):
                      log.eventstolog.discard(toks[3] + " " + toks[4] + " " + phase)
                      log.write("don't show " + toks[3] + " " + toks[4] + " " + phase)
                    else:
                      spoutput(phase + " is not a known phase")
                  else:
                    for phase in log.phasestolog:
                      log.eventstolog.discard(toks[3] + " " + toks[4] + " " + phase)
                      log.write("don't show " + toks[3] + " " + toks[4] + " " + phase)
                else:
                  spoutput(toks[4] + " is not a known data type. try probes echoes or activations")
              else:
                spoutput("You didn't provide a data type, try probes echoes or activations")
            else:
              spoutput(toks[3] + " is not a known memory type. try order syntagmatic or sp")
          else:
            spoutput("You didn't provide a memory type, try order syntagmatic or sp")
      elif toks[1] == "reload" or toks[1] == "reset":
        load()

tokmappings = {"one": "1", "two": "2", "plus": "+", "equals": "=", "minus": "*"}

def myindex(lst, elements):
    '''
    find the earliest occurrance of any of the elements in lst.

    '''

    minimum = maxsize
    for e in elements:
      try:
        m = lst.index(e)
        if m < minimum:
          minimum = m
      except:
        pass
    if minimum == maxsize:
      raise Exception("None of the values were found.")
    else:
      return minimum
        
def processInput(inp):
    global rawcorpus
    #inp = spinput()
    inp = inp.lower()
    toks = inp.split()
    toks = [tokmappings[t] if t in tokmappings else t for t in toks] + ["#"]
    log.write("Input: "+ " ".join(toks))
    if toks[0] == "jamie":
      processCommand(toks)
    else:
      #inp = " " + inp + " "
      rawcorpus = toks
      log.phase = "my encoding"
      encodeCorpus()
      log.phase = "your retrieval"
      Finished = False
      speechoutput = ""
      #totalLength = 0
      while not Finished:
        suffix = ["_"] * 2
        while not Finished and len(suffix) < 5:
          suffix += ["_"]
          toks = corpus + suffix
          #log.write( "Probe: "+ " ".join(toks))
          retrieved = retrieve(toks)
          #log.write("Retrieved: " + " ".join(retrieved))
          print("Retrieved: " +  " ".join(retrieved))
          firstoutput = -len(suffix) - 1
          output = retrieved[firstoutput:]
          print( "Output: "+ " ".join(output))
          Finished = "_" in output or "#" in output 
        if Finished:
          i = myindex(output, ["_", "#"])
          rawcorpus += output[firstoutput:i]
        else:
          rawcorpus += output
          #totalLength += len(output)
        if speechoutput == "":
          speechoutput += " ".join(rawcorpus)
        else:
          speechoutput += " " + " ".join(rawcorpus)
        log.phase = "your encoding"
        encodeCorpus()
        log.phase = "your retrieval"
      #log.write("Output: "+ speechoutput)
      rawcorpus += "_ _ _".split()
      log.phase = "your encoding"
      encodeCorpus()
          
      spoutput(speechoutput)

#conversation()
