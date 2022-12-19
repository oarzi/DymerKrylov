import dimers_sim

def main():
    d_sim = [115, 95, 75, 50]
    L_sim = 120
    times_sim = 6000
    nums_sim = 1500000
    d_procs_sim = 4
    nums_subprocs_sim = 30
    
    simulator = dimers_sim.Simulator(local = True,L=L_sim, times=times_sim, d=d_sim, nums=nums_sim, d_procs_num=d_procs_sim, batch_subprocs_num = nums_subprocs_sim)
    
    simulator.parallel_analysis()
    dimers_sim.plot_analysis(simulator.analysis_rhos, L_sim, times_sim, nums_sim, save=True)
    
if __name__ == '__main__':
	main()

