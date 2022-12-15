import dimers_sim

def main():
    d_sim = [70, 95, 120, 145]
    L_sim = 150
    times_sim = 3000
    nums_sim = 999990
    d_procs_sim = 4
    nums_subprocs_sim = 30
    
    simulator = dimers_sim.Simulator(local = True,L=L_sim, times=times_sim, d=d_sim, nums=nums_sim, d_procs_num=d_procs_sim, batch_subprocs_num = nums_subprocs_sim)
    
    simulator.parallel_analysis()
    dimers_sim.plot_analysis(simulator.analysis_rhos, L_sim, times_sim, nums_sim, save=True)
    
if __name__ == '__main__':
	main()

