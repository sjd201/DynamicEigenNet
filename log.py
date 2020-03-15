logfile = open("sp.log", "w+")

phasestolog = set(["my encoding", "your encoding",  "your retrieval"])
phase = ""

def isPhase(s):
   return s in phasestolog

memorytypestolog = set(["order", "syntagmatic", "sp"])

def isMemoryType(s):
   return s in memorytypestolog

datatypestolog = set(["activations", "echoes", "probes", "probe strings"])

def isDataType(s):
   return s in datatypestolog


eventstolog = set(["order echoes your retrieval", "order probes your retrieval", "order activations your retrieval", "sp echoes your retrieval"])

def write(s, memorytype = "", datatype = ""):
    if memorytype[0:2] == "SP": # allow all SP memories to be logged regardless of the number
      memorytype = "sp"
    if (memorytype + " " + datatype + " " +  phase in eventstolog) or (memorytype == "" and datatype == ""):
      logfile.write(s + "\n")
      logfile.flush()
    

