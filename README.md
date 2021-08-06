# gpu-saving

 Solves consumption saving problem on a GPU

 An agent lives 10 periods. He has to allocate consumption and savings over time to optimize his intertemporal utilty. He recieves a random income over time wich follows a AR(1) in logs.

 We solve this by backward induction over a discrete grid of assets choices and productivity states.

 We take advantage of numba+CUDA package to boost the computations times

 References:
 XX
 XX
 XX

