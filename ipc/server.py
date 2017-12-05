import time
import mmap
import os
import sys
import numpy as np

import posix_ipc as ipc

if len(sys.argv) != 2 :
    print("Usage: %s batch-size" % sys.argv[0])
    sys.exit(-1)

bsize = int(sys.argv[1])
minibatch_num = 2
minibatch_size = bsize // minibatch_num
print("Mini batch size: ", minibatch_size)

if bsize % minibatch_num != 0:
    print("Error: batch-size%minibatch_num!=0")
    sys.exit(-2)

name = "smlee"
bs   = 4*18*19*19
def createSMP(name):
    smp= ipc.Semaphore(name, ipc.O_CREAT)
    smp.unlink()
    return ipc.Semaphore(name, ipc.O_CREAT)

sm = ipc.SharedMemory( name, flags = ipc.O_CREAT, size = 1 + bsize + bs*bsize + 8  + bsize*4*(19*19+2))

smp_counter =  createSMP("lee_counter")


smpA = []
smpB = []
for i in range(bsize):
    smpA.append(createSMP("lee_A_%d" % i))
    smpB.append(createSMP("lee_B_%d" % i))

# memory layout of sm:
# counter |  ....... | ....... | ....... |
#

mem = mmap.mmap(sm.fd, sm.size)
sm.close_fd()

mv  = np.frombuffer(mem, dtype=np.uint8, count= 1 + bsize + bs*bsize + 8  + bsize*4*(19*19+2))
counter = mv[0:1+bsize]
inp     = mv[  1+bsize:1+bsize + bs*bsize]
memout =  mv[          1+bsize + bs*bsize + 8:]

import nn

counter[0] = bsize
for i in range(bsize):
    counter[1 + i ] = 0

smp_counter.release()

# waiting clients to connect
print("Waiting for %d autogtp instances to run" % bsize)
# for i in range(bsize):
#     smpB[i].acquire()
#
# print("OK Go!")

# now all clients connected
batch_input_size = bs // 4 * minibatch_size
dt = np.zeros( batch_input_size, dtype=np.float32)

#net = nn.net
batch_output_size = minibatch_size*(19*19+2)
npout = np.zeros ( batch_output_size )
import gc

while True:
    # print(c)

    for iminibatch in range(minibatch_num):
        ibatch_start = iminibatch*minibatch_size
        ibatch_end   = (iminibatch+1)*minibatch_size
        
        # wait for data
        for i in range(ibatch_start, ibatch_end):
            smpB[i].acquire()

        dt[:] = np.frombuffer(inp[batch_input_size*4*iminibatch:batch_input_size*4*(iminibatch+1)], dtype=np.float32, count=batch_input_size)

        nn.netlock.acquire(True)   # BLOCK HERE
        if nn.newNetWeight != None:
            nn.net = None
            gc.collect()  # hope that GPU memory is freed, not sure :-()
            weights, numBlocks, numFilters = nn.newNetWeight
            print(" %d channels and %d blocks" % (numFilters, numBlocks) )
            nn.net = nn.LZN(weights, numBlocks, numFilters, minibatch_size)
            net = nn.net
            print("...updated weight!")
            nn.newNetWeight = None
        nn.netlock.release()

        net[0].set_value( dt.reshape( (minibatch_size, 18, 19, 19) ) )

        qqq = net[1]().astype(np.float32)
        ttt = qqq.reshape(minibatch_size * (19*19+2))
        #print(len(ttt)*4, len(memout))
        memout[batch_output_size*4*iminibatch:batch_output_size*4*(iminibatch+1)] = ttt.view(dtype=np.uint8)

        for i in range(ibatch_start, ibatch_end):
            smpA[i].release() # send result to client

