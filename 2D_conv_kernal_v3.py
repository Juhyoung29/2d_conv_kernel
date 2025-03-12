import allo
from allo.ir.types import float32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

### convolution kernal with systolic array (basic) ###

IR, IC = 5, 5 # input column and row
FR, FC = 3, 3 # filter column and row
OR, OC = 3, 3 # output column and row
P0, P1 = OR*OC + 2, 3 # we need a PE per element in the output matrix, we aso need two layers of PE to add up results

@df.region()
def top():
    fifo_A = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))
    fifo_B = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))

    @df.kernel(mapping = [P0, P1])

    # let A be the input matrix, B be the filter matrix 
    # and C be the output matrix. For this simple convolution
    # kernel we will be implementing a Valid kernel

    def conv_kernel(A: float32[IR, IC], B: float32[FR, FC]):
        pi, pj = df.get_pid()

        # we do not use these PEs
        with allo.meta_if(pi in {0, P0 - 1} and pj in {0, P1 - 1}):
            pass

        # this meta_if loads in the input matrix into the PEs
        with allo.meta_elif(pj == 0):                  
            for row,col in allo.grid(FR, FC):
                with allo.meta_if(pi <= 3):
                    fifo_A[pi,pj].put(A[(pi - 1) + row, col])
                with allo.meta_elif(pi <= 6):
                    fifo_A[pi,pj].put(A[(pi - 1) + row, col + 1])
                with allo.meta_else:
                    fifo_A[pi,pj].put(A[(pi - 1) + row, col + 2])
        
        #this meta_elif loads in the filter matrix into the PEs
        with allo.meta_elif(pi == 0):
            for row,col in allo.grid(FR, FC):
                fifo_B[pi,pj].put(B[FR - row - 1, FC - col - 1]) # we do this becasue we want the last entry of the filter matrix to go first into the PE

        # this meta_elif loads the partial sums into the drain PEs
        with allo.meta_elif(pj == P1 - 1 and pi > 0):
            for row in range(P0):
                drain_A: float32 = fifo_A[pi,pj].get()
        with allo.meta_elif(pi == P0 - 1 and pj > 0):
            for col in range(P1 - 1): # for this implementation this only runs once
                drain_B: float32 = fifo_B[pi,pj].get()
        
        # this meta_else does the main multiplication of the convolution kernel
        with allo.meta_else():
            c: float32 = 0
            for k in range(OR*OC):
                a: float32 = fifo_A[pi,pj].get()
                b: float32 = fifo_B[pi,pj].get()
                c += a*b 
                fifo_A[pi, pj + 1].put(c)
                fifo_B[pi + 1,pj].put(a)


    @df.kernel(mapping=[P0,P1-1])
    def adder(C: float32[OR, OC]): # 9 by 1 becasue I need 9 PE in a column to add up the 9 outputs from
                                   # the systolic array
        for i in range(OC*OR):
            a: float32 = fifo_A[pi,pj+i].get()
            #b: float32 = fifo_B[pi,pj+i].get()
            with allo.meta_if(i < 3):
                C[1, i] += a
            with allo.meta_elif(i < 6):
                C[2, i-3] += a
            with allo.meta_else:
                C[3, i-6] += a


### Base convolution kernel (Truth that we compare against) ###

#def test_conv2D_lb():
def conv2D_lb(A: int32[IR, IC]) -> int32[OR, OC]:
    B: int32[OR, OC] = 0
    for y, x in allo.grid(OR, OC): # these are the output dimensions
        v: int32 = 0
        for r, c in allo.reduction(FR, FC): #this is the filter dimensions
            v += A[y + r, x + c]
        B[y, x] = v
    return B




### testing the systolic convolution kernel ###

def test_convolution():
    s = allo.customize(conv2D_lb)
    LB = s.reuse_at(s.A, "y")
    print(s.module)
    test_mod = s.build()

    A_sys = np.random.rand(IR, IC).astype(np.float32)
    B_sys = np.random.rand(FR, FC).astype(np.float32)
    C_sys = np.zeros((OR, OC), dtype = np.float32)

    sim_mod = df.build(top, target = "simulator")
    sim_mod(A_sys, B_sys, C_sys)
    np.testing.assert_allclose(C_sys, test_mod, atol=1e-5)
    print("simulation passed!")

    # mod = df.build(top)
    # if hls.is_available("vitis_hls"):
    #     C = np.zeros((IR,IC), dtype = np.float32)
    #     mod(A, B, C)
    #     np.testing.assert_allclose(C, test_mod, atol = 1e-5)