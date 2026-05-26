import pandas as pd
import struct
import re


def read_file(fname, fmt = 'dicccc', size = 16):
  with open(fname, 'rb') as f:
    data = []
    chunk = f.read(size)
    data.append(chunk)
    while chunk != "":
        print(len(chunk))
        print(struct.unpack(fmt, chunk))
        chunk = f.read(size)
        data.append(chunk)
    print("data.len = {} for fname {}".format(data, fname))
    return data


def read_drrun(fname, outfile="trace.csv"):
    i, data = 0, []
    with open(fname, 'r') as f:
        line, tid, tcount, w, window = " ", None, 0, [], 10
        while len(line) > 0: #and i < 1000:
            line = f.readline()
            if re.match('\s+[0-9]+\s+[0-9]+:\s+[0-9]+\s+.*', line):
                toks = re.split('\s+', line)
                #print("line = {}".format(line))
                if len(w) >= window: 
                    w.pop(0)
                if "ifetch" in line:
                    w.append(1)
                elif "read" in line or "write" in line:
                    w.append(2)
                if not tid == toks[3]:
                    tid, cpu, mem, tsz, tcount = toks[3], 0, 0, 0, tcount+1
                cpu = cpu + 1 if "ifetch" in line else cpu 
                mem = mem + 1 if "read" in line or "write" in line else mem
                tsz = tsz + 1
                util = lambda v: sum([1 if i==v else 0 for i in w])/len(w)
                if i % 100 == 0:
                    print("i={}, tid={}, cpu={}, mem={}, tsz={}".format(
                        i,tid,util(1),util(2),tsz))
                    data.append([i,tid,util(1),util(2),tsz])
            i = i + 1
        pd.DataFrame(data).to_csv(outfile,header=["i","tid","cpu","mem","tsz"])
        print("tcount = {}, output in {}".format(tcount, outfile))


if __name__ == "__main__":
    #read_file('charlie_trace-1_17571657100049929577.1006509.memtrace', 
    #          'dixxxxcxxxxxxx', 24)
    read_drrun('out.txt')
