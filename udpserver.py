import sys
import glob
import gzip
import random
import math
import multiprocessing as mp
from theano import *
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import relu
from theano.tensor.nnet.bn import batch_normalization_test as bn
import numpy as np
import trollius

if len(sys.argv) != 3:
    print("Usage: %s [weight-file] [batch-size]" % sys.argv[0])
    sys.exit(-1)

filename = sys.argv[1]
QLEN     = int(sys.argv[2])

print("Leela Zero UDP Neural Net Service")

def loadWeightFile(filename):
    f = open(filename, "r")

    linecount = 0

    def testShape(s, si):
        t = 1
        for l in s:
            t = t * l
        if t != si:
            print("ERRROR: ", s, t, si)

    FORMAT_VERSION = "1\n"

    print("Detecting the number of residual layers...")

    w = f.readlines()
    linecount = len(w)

    if w[0] != FORMAT_VERSION:
        print("Wrong version")
        sys.exit(-1)

    count = len(w[2].split(" "))
    print("%d channels..." % count)

    residual_blocks = linecount - (1 + 4 + 14)

    if residual_blocks % 8 != 0:
        print("Inconsistent number of layers.")
        sys.exit(-1)

    residual_blocks = residual_blocks // 8
    print("%d blocks..." % residual_blocks)

    plain_conv_layers = 1 + (residual_blocks * 2)
    plain_conv_wts = plain_conv_layers * 4


    weights = [ [float(t) for t in l.split(" ")] for l in w[1:] ]
    return (weights, residual_blocks, count)


weights, numBlocks, numFilters = loadWeightFile(filename)


def LZN(ws, nb, nf):
    # ws: weights
    # nb: number of blocks
    # nf: number of filters

    global wc  # weight counter
    wc = -1

    def loadW():
        global wc
        wc = wc + 1
        return ws[wc]


    def mybn(inp, nf, params, name):
        #mean0 = T.vector(name + "_mean")
        w = np.asarray(loadW(), dtype=np.float32).reshape( (nf) )
        mean0 = shared(w)
        # params.append(In(mean0, value=w))

        #var0  = T.vector(name + "_var")
        w = np.asarray(loadW(), dtype=np.float32).reshape( (nf) )
        var0 = shared(w)
        #params.append(In(var0, value=w))

        bn0   = bn(inp, gamma=T.ones(nf), beta=T.zeros(nf), mean=mean0,
                var=var0, axes = 'spatial', epsilon=1.0000001e-5)

        return bn0

    def myconv(inp, inc, outc, kernel_size, params, name):
        #f0 = T.tensor4(name + '_filter')
        w = np.asarray(loadW(), dtype=np.float32).reshape( (outc, inc, kernel_size, kernel_size) )
        #params.append(In(f0, value=w))
        f0 = shared(w)

        conv0 = conv2d(inp, f0, input_shape=(None, inc, 19, 19),
                border_mode='half', filter_flip=False, filter_shape=(outc,inc,
                    kernel_size,kernel_size))
        b = loadW()  # zero bias
        return conv0

    def residualBlock(inp, nf, params, name):
        conv0 = myconv(inp, nf, nf, 3, params, name + "_conv0")
        bn0   = mybn(conv0, nf, params, name + "_bn0")
        relu0 = relu(bn0)

        conv1 = myconv(relu0, nf, nf, 3, params, name + "_conv1")
        bn1   = mybn(conv1, nf, params, name + "_bn1")

        sum0  = inp + bn1
        out   = relu(sum0)

        return out

    def myfc(inp, insize, outsize, params, name):
        # W0 = T.matrix(name + '_W')
        w = np.asarray(loadW(), dtype=np.float32).reshape( (outsize, insize) ).T
        #params.append(In(W0, value=w))
        W0 = shared(w)

        # b0 = T.vector(name + '_b')
        b = np.asarray(loadW(), dtype=np.float32).reshape( (outsize) )
        # params.append(In(b0, value=b))
        b0 = shared(b)

        out = tensor.dot(inp, W0) + b0
        return out


    params = []
    x = T.tensor4('input')
    params.append(x)
    conv0 = myconv(x, 18, nf, 3, params, "conv0")
    bn0   = mybn(conv0, nf, params, "bn0")

    relu0 = relu(bn0)
    inp = relu0

    for i in range(nb):
        inp = residualBlock(inp, nf, params, "res%d" % (i+1))

    polconv0 = myconv(inp, nf, 2, 1, params, "polconv0")
    polbn0   = mybn(polconv0, 2, params, "polbn0")
    polrelu0 = relu(polbn0)
    polfcinp = polrelu0.flatten(ndim=2)
    polfcout = myfc(polfcinp, 19*19*2, 19*19+1, params, "polfc")

    out = polfcout

    valconv0 = myconv(inp, nf, 1, 1, params, "valconv0")
    valbn0   = mybn(valconv0, 1, params, "valbn0")
    valrelu0 = relu(valbn0)

    valfc0inp = valrelu0.flatten(ndim=2)
    valfc0out = myfc(valfc0inp, 19*19, 256, params, "valfc0")
    valrelu0  = relu(valfc0out)

    valfc1out = myfc(valrelu0, 256, 1, params, "valfc1")
    valout  = valfc1out

    out = T.concatenate( [polfcout, T.tanh(valout)], axis=1 )
    return function(params, out)


net = LZN(weights, numBlocks, numFilters)

### For testing purpose
inp1 = [math.sin(i) for i in range(1*19*19*18) ]
inp2 = [math.cos(i) for i in range(1*19*19*18) ]
inp = []
inp.extend(inp1)
inp.extend(inp2)
inpp = np.asarray(inp, dtype=np.float32).reshape( (2,18,19,19) )
#print(net(inpp))
#for i in range(1000):
#    net(inpp)


class UDPServerProtocol:
    def __init__(self, QLEN=QLEN):      # QLEN === BATCH_SIZE
        self.inp = np.zeros( (QLEN, 19*19*18), dtype=np.float32)
        self.clientAddrs = list(range(QLEN))
        self.counter = 0
        self.QLEN = QLEN

        print("Setting batch-size = %d" % self.QLEN)

    def connection_made(self, transport):
        self.transport = transport

    def computeForward(self, ):
        self.counter = 0
        myinp = self.inp.reshape( (self.QLEN, 18, 19, 19) )
        out = net(myinp).astype(np.float32)


        ports = []      # never sent to an address twice each batch
        for i in range(self.QLEN):
            #print(out[0,0].data.tobytes())
            if not self.clientAddrs[i][1] in ports:
                self.transport.sendto(out[i, :].data, self.clientAddrs[i] )
                ports.append(self.clientAddrs[i][1])
            #self.transport.sendto(b"1234", self.ADDRS[i] )


    def datagram_received(self, data, addr):
        if len(data) != 19*19*18*4:
            print("ERR")
        try:
            self.inp[self.counter, :] = np.frombuffer(data, dtype=np.float32, count = 19*19*18)
            self.clientAddrs[self.counter] = addr

            # self.transport.sendto(data, addr)
            self.counter = self.counter + 1
            if self.counter == self.QLEN:
                self.computeForward()
        except Exception as ex:
            int(ex)

loop = trollius.get_event_loop()
print("Starting UDP server at 127.0.0.1:9999")
# One protocol instance will be created to serve all client requests
listen = loop.create_datagram_endpoint(
    UDPServerProtocol, local_addr=('127.0.0.1', 9999))
transport, protocol = loop.run_until_complete(listen)

try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

transport.close()
loop.close()


