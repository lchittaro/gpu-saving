# multiple GPU computing based on
# https://link.springer.com/chapter/10.1007/978-3-030-71593-9_16
# https://colab.research.google.com/github/evaneschneider/parallel-programming/blob/master/COMPASS_gpu_intro.ipynb
# and adams code


#--------------------------------#
#    0. Housekeeping             #
#--------------------------------#

# set cuda simulator to enable debbug
# always BEFORE loading numba
# import os
# os.environ['NUMBA_ENABLE_CUDASIM'] = "1"

from numba import cuda, float32
import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt
import math







#--------------------------------#
#   1. Define grids
#--------------------------------#

def create_gpu_input(nx, ne, T):
    '''
    Creates a 1D vector that saves
      - all parameters of cs problem
      - all entries of the value function
    this vector (concat) will feed the VFI kernel
    
    inputs:
      -nx: elements of asset grid
    
    output:
      -concat_gpu: a vector allocated in gpu memory
    '''
    
    
    
    # Grid of assets
    ## Define parameters
    xmin            = 0.1
    xmax            = 4.0
    
    
    ## Create grid of assets (xgrid)
    xgrid = np.zeros(nx)
    size = nx
    xstep = (xmax - xmin) /(size - 1)
    for i in range(nx):
        xgrid[i] = xmin + i *xstep
    
    
    # Grid of productivity
    ## Define parameters
    m              = 1.5
    sigma_eps      = 0.02058
    lambda_eps     = 0.99
    
    
    ## Create grid of productivity (egrid)
    egrid = np.zeros(ne)
    sigma_y = np.sqrt((sigma_eps ** 2) / (1 - (lambda_eps ** 2)))
    estep = 2*sigma_y*m / (ne-1)
    for i in range(ne):
        egrid[i] =  -m * np.sqrt( (sigma_eps ** 2) / (1 - (lambda_eps ** 2)) ) + i * estep
    
    
    # Transition probability matrix (P) Tauchen (1986)
    ## Create grid of transtion
    P     = np.zeros((ne, ne))
    mm    = egrid[1] - egrid[0]
    
    # j is current productivity state
    for j in range(ne): 
        # k is future productivity state
        for k in range(ne):
            if k == 0:
                P[j, k] = norm.cdf((egrid[k] - lambda_eps*egrid[j] + (mm/2))/sigma_eps)
            elif k == ne-1:
                P[j, k] = 1 - norm.cdf((egrid[k] - lambda_eps*egrid[j] - (mm/2))/sigma_eps)
            else:
                P[j, k] = norm.cdf((egrid[k] - lambda_eps*egrid[j] + (mm/2))/sigma_eps) - norm.cdf((egrid[k] - lambda_eps*egrid[j] - (mm/2))/sigma_eps)
    
    ## vectorize P
    P_vec = P.reshape((ne*ne), order='C')
    
    ## Exponential of the grid e
    for i in range(ne):
        egrid[i] = np.exp(egrid[i])
    
    
    
    # Initialize value funcion
    Vtemp = np.zeros(nx*ne*T)
    
    
    # 1-D vector with all the inputs for VFI
    concat = np.concatenate((xgrid, egrid, P_vec, Vtemp, Vtemp), axis = 0) #needs double parentesis
    concat = concat.astype('float32')
    
    # creates an empty vectorin the managed memory (aka unified memory) and
    # allocates there the parameters and the entries of the value and policy
    concat_gpu = cuda.managed_array(len(concat), dtype=np.float32, attach_global=True)
    for i in range(len(concat)):
        concat_gpu[i] = concat[i]
    
    return concat_gpu
    
    
    
    
    
#--------------------------------#
#   2. Create a kernel
#--------------------------------#

# Kernel to compute Value given age, grids and transition matrix and horizont T.
@cuda.jit
def value_function(concat, age, completed):
    
    '''
    Given an age and 1D vector that contains all parameters of the problem,
    computes the value function and the maximizer and save it in the same
    1D vector
    
    inputs:
        -age: age of the agent, must be passed backwards
        -concat: contains asset and income grid, transition matrix, and
            spare slots to save value and policy functions
    
    output:
      -concat_gpu: same as before, but now it has the computation of the value
        and policy function
    '''
    
    # Compute flattened index inside the array
    ind = int(cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x) + completed

    # grid parameters
    nx     = int(50000)
    ne     = int(15)
    
    if (ind<(nx*ne)):
    
        ix        =  int(math.floor(ind/ne))          # incoming asset level 
        ie        =  int(ind - ix*ne)                 # incoming productivity
        VV        =  float32(-math.pow(10.0,3))       # to store maximum value
        maximizer =  float32(0.0)                     # to store maximizer
        utility   =  float32(0.0)                      # initializes utility
        ixp       = int(0)                            # will be used to loop over
                                                      # possible values of asset
        
        
        if (age<9):                                   # 9 is terminal period, 0 is initial
            
            while (ixp<nx):                           # starts looping decissions
                
                iep = int(0)
                expected_value = float32(0.0)
            
                while (iep<ne):                       # loop over exogenous states
                                                      # to get the expected value
                
                    expected_value = expected_value + concat[nx + ne + ne*ie + iep] *  concat[nx + ne + ne*ne + (age+1)*nx*ne + ixp*ne + iep]
                    iep += 1
                    
                
                # get consumption & utility given a incoming state (asset and productivity) and a decission
                cons    =  5.0 * concat[nx + ie] + 1.07 * concat[ix] - concat[ixp]
                utility = - (1.0 / cons) + 0.97 * expected_value

                # applies non-negativity constraint
                if (cons<0.0):
                    utility = -10.0**5
    
                # compares to last maximum
                if utility > VV:
                    VV = utility
                    maximizer = concat[ixp]
                
                # removes utility and moves to the next potential decission
                utility   = float32(0.0)
                ixp +=1

            # saves maximum utility for a given incoming asset level
            concat[nx + ne + ne*ne + age*ne*nx + ind] = VV
            concat[nx + ne + ne*ne + 10*ne*nx + age*ne*nx + ind] = maximizer
        
        
        
        else:                                      #if age is the terminal period
            
            cons = 5.0 * concat[nx + ie] + 1.07 * concat[ix] - concat[0] # agent stays with the min of the grid in afterlife
            utility = -(1.0/cons)
        
            if(cons < 0.0):
                utility = -math.pow(10.0,5)
      
            concat[nx + ne+ ne*ne + age*ne*nx + ind]            = utility
            concat[nx + ne+ ne*ne + 10*nx*ne + age*ne*nx + ind] = concat[0];

        


#--------------------------------#
# 3. Solve cs prob for a given gridsize
#--------------------------------#

# these parameters should be the same that sections 1 and 2
nx = 50000
ne = 15
T = 10
nparams = nx + ne + ne*ne 



print('\n\n======================================')

print('CS problem at multiple gpu, grid size of 50,000 ')
print('\n----------------')
print('Start:', time.ctime(time.time()))
print('\n----------------')

print('grid size :', nx)
print('\n----------------')


print('Creating gpu input...')

# allocates all setup in gpu, using a grid of 50000
concat_gpu = create_gpu_input(nx, ne, T)
print(concat_gpu[(nparams+ nx*ne*T):(nparams+ nx*ne*T + 30)])
print(concat_gpu[nparams:(nparams+30)])

print('\n----------------')

print('Ended at: ', time.ctime(time.time()))
print('\n----------------')

nparams = nx + ne + ne*ne 


print('Computing VFI (10 repetitions)...')

# define usage gpus, cells, blocks & threads
gpus = len(cuda.gpus)
cellspergpu = math.ceil(nx*ne/gpus)
threadsperblock = 1024
blockspergpu = math.ceil(cellspergpu/threadsperblock)


# times will be saved in this array
times = np.zeros(T)

# Compute 10 times (lap) a 10 period long CS problem using the allocated vector
# first lap is only to cook the kernel

for lap in range(10):
    start = time.time()
    
    for age in range(T-1, -1, -1):
        
        
        # send instructions to all devices
        for gpu_idx in range(len(cuda.gpus)):
            cuda.select_device(gpu_idx)
            value_function[blockspergpu, threadsperblock](concat_gpu, age, gpu_idx*cellspergpu)

        
        # syncronize all devices
        for gpu_idx in range(len(cuda.gpus)):
            cuda.select_device(gpu_idx)
            cuda.synchronize()        
            
        
    # end time measurement, after GPU finished computing all                  
    finish = time.time()
    times[lap] = finish - start

print('\n----------------')


print('Ended at: ', time.ctime(time.time()))
print('\n----------------')
print('Results')
print('Computation times:')
print(times)
print('Average: ')
print(np.mean(times[1:T]))
print('\n----------------')

# allocate gpu vector in cpu

out = np.zeros(nx + ne + ne*ne + T*nx*ne*2)
for i in range(len(out)):
    out[i] = concat_gpu[i]
    



print('First entries of VF:')
print(out[nparams:(nparams+30)])
print('\n----------------')

print('First entries of policy:')
print(out[(nparams + 10*nx*ne):(nparams + 10*nx*ne+30)])
print('\n----------------')
print('End of program')
print('\n=========================================')
print('\n=========================================')