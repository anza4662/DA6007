import sys

textin = sys.stdin.readlines()
textout = ""
textin = [line.split("\n")[0] for line in textin]

for line in textin:
    line = '"' + line + '",\n'
    textout += line

print(textout)
