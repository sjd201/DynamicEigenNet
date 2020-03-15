from sys import stdin

for line in stdin:
  toks = line.split()
  for tok in toks:
    print(tok.lower())
