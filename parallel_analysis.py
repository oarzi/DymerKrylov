import dimers_sim
import time

def main():
    d_sim = [115, 95, 75, 55]
    L_sim = 120
    times_sim = 2000
    nums_sim = 200000
    d_procs_sim = 4
    nums_subprocs_sim = 8
    
    simulators = [dimers_sim.Simulator(local = False, L=L_sim, times=times_sim, d=d, nums=nums_sim, batch_subprocs_num = 2) for d in d_sim]
    
    results =  dimers_sim.Simulator.simulate_parallel(simulators, d_procs_sim)
    file_name = 'experiment_L{}_t{}_n{}____{}'.format(L_sim, times_sim, nums_sim, time.strftime("%Y_%m_%d__%H_%M"))
    title = "L{}, times={}, nums={}".format(L_sim, times_sim, nums_sim)
    dimers_sim.plot_analyses(results, title=title, save=True, name = file_name)
    
    experiment = dimers_sim.Experiment(file_name,
                                      "analyses",
                                      results,
                                      description='Paths over L=120 for various values of iniital position')
    
    experiment.save()                      
if __name__ == '__main__':
	main()

