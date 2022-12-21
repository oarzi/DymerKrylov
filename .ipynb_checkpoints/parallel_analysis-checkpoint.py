import dimers_sim
import time

def varying_batch_size(L_sim, times_sim, d_sim, batch_size, procs_sim, batch_procs_num):
    # d_sim = [115]
    # L_sim = 120
    # times_sim = 2000
    # batch_size = 200000
    # d_procs_sim = 4
    # batch_procs_num = 8
    if type(batch_procs_num) == list and len(batch_procs_num) < len(batch_size):
        batch_procs_num = batch_procs_num + (len(batch_size) - len(batch_procs_num))*batch_procs_num[-1:]
    dir_name = "analyses/batch_size"
    
    simulators = [dimers_sim.Simulator(local = True, L=L_sim, times=times_sim, d=d_sim, batch=b, batch_procs_num = bn, dir_name=dir_name) for b, bn in zip(batch_size, batch_procs_num)]
    
    results =  dimers_sim.Simulator.simulate_parallel(simulators, procs_sim)
    file_name = 'experiment_L{}_t{}_b{}____{}'.format(L_sim, times_sim, batch_size, time.strftime("%Y_%m_%d__%H_%M"))
    title = "Evolution for varying batch size - L={}, # times={}, d={}".format(L_sim, times_sim, d_sim)
    
    experiment = dimers_sim.Experiment(file_name,
                                      "analyses/varying_batch_size/",
                                      results,
                                      description='Varying batch size experiment for L={}, times={}, d={}, batch_size={}'.format(L_sim, times_sim, d_sim, batch_size))
    
    experiment.save() 
    dimers_sim.plot_analyses(results,label = 'batch', title=title, save=True, name = "varying_batch_size/" + file_name)

                         
if __name__ == '__main__':
    

    args = dimers_sim.get_experiment_args()
    
    Experiments = {'bs' : varying_batch_size}
    experiment_execute = Experiments[args.experiment]
    experiment_execute(args.L[0], args.times[0], args.d[0], args.batch, args.procs_sim[0], args.batch_procs)
    # varying_batch_size()

