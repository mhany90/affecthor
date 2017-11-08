import sys

file = sys.argv[1]
fileread = open(file, 'r')
filenew = open(file+".strip",'w')
lines = fileread.readlines()
striplines = [i.replace('"', '') for i in lines]
for line in striplines:
    filenew.write(line)
fileread.close()
filenew.close()
