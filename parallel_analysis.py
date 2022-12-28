import dimers_sim
import time

def varying_batch_size(L_sim, times_sim, d_sim, batch_size, procs_sim, batch_procs_num, name):
    print("--- ===   varying_batch_size   === ---")
    if type(batch_procs_num) == list and len(batch_procs_num) < len(batch_size):
        batch_procs_num = batch_procs_num + (len(batch_size) - len(batch_procs_num))*batch_procs_num[-1:]
    d_sim = d_sim[0]
    dir_name = "varying_batch_size/"
    
    file_name = name + '_bs_experiment_L{}_t{}_b{}____'.format(L_sim, times_sim, batch_size,
                                                      )
    
    title = "Evolution for varying batch size - L={}, # times={}, d={}".format(L_sim, times_sim, d_sim)
    
    
    simulators = [dimers_sim.Simulator(local = False, L=L_sim, times=times_sim, d=d_sim, batch=b, batch_procs_num = bn, dir_name=dir_name) for b, bn in zip(batch_size, batch_procs_num)]
    
    results =  dimers_sim.Simulator.simulate_parallel(simulators, procs_sim)


    experiment = dimers_sim.Experiment(file_name + time.strftime("%Y_%m_%d__%H_%M"),
                                      "analyses/" + dir_name,
                                      results,
                                      description='Varying batch size experiment for L={}, times={}, d={}, batch_size={}'.format(L_sim, times_sim, d_sim, batch_size))
    
    experiment.save() 
    dimers_sim.plot_analyses(results,label = 'batch', title=title, save=True, name = dir_name + file_name  + 
                             time.strftime("%Y_%m_%d__%H_%M"))
    
def varying_initial_conditions(L_sim, times_sim, d_sim, batch_size, procs_sim, batch_procs_num, name):
    print("--- ===   varying_initial_conditions   === ---")
    if len(batch_procs_num) < len(d_sim):
        batch_procs_num = batch_procs_num + (len(d_sim) - len(batch_procs_num))*batch_procs_num[-1:]
    batch_size = batch_size[0]
    dir_name = "varying_initial_conditions/"
    file_name = name + '_ic_experiment_L{}_t{}_d{}____'.format(L_sim, times_sim, d_sim)
    title = "Evolution for initial position - L={}, # times={}, d={}".format(L_sim, times_sim, d_sim)
    
    print("batch_procs_num={}".format(batch_procs_num))
    
    simulators = [dimers_sim.Simulator(local = False, L=L_sim, times=times_sim, d=d, batch=batch_size, batch_procs_num = bn, dir_name=dir_name) for d, bn in zip(d_sim, batch_procs_num)]
    
    results =  dimers_sim.Simulator.simulate_parallel(simulators, procs_sim)

    
    experiment = dimers_sim.Experiment(file_name +  time.strftime("%Y_%m_%d__%H_%M"),
                                      "analyses/" + dir_name,
                                      results,
                                      description='Varying initial position size experiment for L={}, times={}, d={}, batch_size={}'.format(L_sim, times_sim, d_sim, batch_size))
    
    experiment.save() 
    dimers_sim.plot_analyses(results, label = 'd', title=title, save=True, name = dir_name + file_name + 
                             time.strftime("%Y_%m_%d__%H_%M"))

                         
if __name__ == '__main__':
    parser = dimers_sim.get_experiment_args()
    args = parser.parse_args()
    print(args)
    Experiments = {'bs' : varying_batch_size, "ic" : varying_initial_conditions}
    experiment_execute = Experiments[args.experiment]
    experiment_execute(args.L[0], args.times[0], args.d, args.batch, args.procs_sim[0], args.batch_procs, 
                       args.name)

