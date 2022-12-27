import dimers_sim
import pickle
import time

def main():
    d_sim = [50]
    L_sim = 120
    times_sim = 4000
    nums_sim = [10000, 50000, 250000, 1000000]
    d_procs_sim = 1
    nums_subprocs_sim = 20
    
    simulators =[dimers_sim.Simulator(local = True,L=L_sim, times=times_sim, d=d_sim, nums=n, d_procs_num=d_procs_sim, batch_subprocs_num = nums_subprocs_sim, save=False) for n in nums_sim]
            
    res = [s.parallel_analysis() for s in simulators]
    
    with open('analyses/same_d__diff_nums/same_d__diff_nums__{}.pickle'.format(time.strftime("%Y_%m_%d__%H_%M")), 'wb') as handle:
        pickle.dump(res, handle)
    
    
if __name__ == '__main__':
	main()

