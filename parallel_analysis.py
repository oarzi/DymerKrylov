import dimers_sim
import dimers_analysis
import time
import sys
import os

def varying_batch_size(args):
    print("--- ===   varying_batch_size   === ---")
    L_sim, times_sim, d_sim, batch_size, procs_sim, batch_procs_num, name = args.L[0], args.times[0], args.d[0], args.batch, args.procs_sim[0], args.batch_procs, args.name
    if type(batch_procs_num) == list and len(batch_procs_num) < len(batch_size):
        batch_procs_num = batch_procs_num + (len(batch_size) - len(batch_procs_num))*batch_procs_num[-1:]

    dir_name = "varying_batch_size/"
    
    file_name = name[0] + '_bs_experiment_L{}_t{}_b{}____'.format(L_sim, times_sim, batch_size)
    
    title = "Evolution for varying batch size - L={}, # times={}, d={}".format(L_sim, times_sim, d_sim)
    
    
    simulators = [dimers_sim.Simulator(local = False, L=L_sim, times=times_sim, d=d_sim, batch=b, batch_procs_num = bn, dir_name=dir_name) for b, bn in zip(batch_size, batch_procs_num)]
    
    results =  dimers_sim.Simulator.simulate_parallel(simulators, procs_sim)


    experiment = dimers_analysis.Experiment(file_name + time.strftime("%Y_%m_%d__%H_%M"),
                                      "analyses/" + dir_name,
                                      results,
                                      description='Varying batch size experiment for L={}, times={}, d={}, batch_size={}'.format(L_sim, times_sim, d_sim, batch_size))
    
    experiment.save() 
    dimers_analysis.plot_analyses(results,label = 'batch', title=title, name = dir_name + file_name  + 
                             time.strftime("%Y_%m_%d__%H_%M"))
    
def varying_initial_conditions(args):
    print("--- ===   varying_initial_conditions   === ---")
    L_sim, times_sim, d_sim, batch_size, procs_sim, batch_procs_num, name = args.L[0], args.times[0], args.d, args.batch[0],  args.procs_sim[0], args.batch_procs, args.name
    if len(batch_procs_num) < len(d_sim):
        batch_procs_num = batch_procs_num + (len(d_sim) - len(batch_procs_num))*batch_procs_num[-1:]

    dir_name = "varying_initial_conditions/"

    file_name = name[0] + '_ic_experiment_L{}_t{}_d{}_b{}___'.format(L_sim, times_sim, d_sim, batch_size)

    title = "Evolution for initial position - L={}, # times={}, d={}".format(L_sim, times_sim, d_sim)
    
    print("batch_procs_num={}".format(batch_procs_num))
    
    simulators = [dimers_sim.Simulator(local = False, L=L_sim, times=times_sim, d=d, batch=batch_size, batch_procs_num = bn, dir_name=dir_name) for d, bn in zip(d_sim, batch_procs_num)]
    
    results =  dimers_sim.Simulator.simulate_parallel(simulators, procs_sim)

    
    #experiment = dimers_analysis.Experiment(file_name +  time.strftime("%Y_%m_%d__%H_%M"),
    #                                  "analyses/" + dir_name,
    #                                 results,
    #                                  description='Varying initial position size experiment for L={}, times={}, d={}, batch_size={}'.format(L_sim, times_sim, d_sim, batch_size))
    
    #experiment.save() 
    dimers_analysis.plot_analyses(results, label = 'd', title=title, name = dir_name + file_name + 
                             time.strftime("%Y_%m_%d__%H_%M"))
                             
def quantum(args):
    print("--- ===   Quantum simulation   === ---")
    L_sim, times_sim, check, d_sim, name, p_gate, from_file = args.L[0], args.times[0], args.check[0], args.d[0], args.name, args.p[0], args.file[0]
    
    dir_name = "quantum/L{}_d{}/".format(L_sim, d_sim)
    
    file_name = name + '_q_experiment_L{}_d{}_p{}'.format(L_sim, d_sim, p_gate)
    
    title = "Evolution for quantum - L={}, # times={}, d={}".format(L_sim, times_sim, d_sim)
    
    
    simulator = dimers_sim.QuantumSimulator(local = False, L=L_sim, times=times_sim, check_interval=check, d=d_sim,
                                            prob=p_gate, file_name=file_name, dir_name="analyses/" + dir_name,
                                            from_file=from_file)
    
    results =  [simulator.simulate()]

    #experiment = dimers_analysis.Experiment(file_name + time.strftime("%Y_%m_%d__%H_%M"),
    #                                  "analyses/" + dir_name,
    #                                  results,
    #                                  description='quantum experiment for L={}, times={}, d={}'.format(L_sim, times_sim, d_sim))
    
    #experiment.save() 
    dimers_analysis.plot_analyses(results,label = 'L', title=title, name = dir_name + file_name  + 
                             time.strftime("%Y_%m_%d__%H_%M"), save=True)
    
def varying_gate_probabilities(args):
    print("--- ===   varying_gate_probabilities   === ---")
    L_sim, times_sim, check, d_sim, p_gate, batch_size, procs_sim, batch_procs_num, name = args.L[0], args.times[0], args.check[0], args.d[0], args.p[0], args.batch[0], args.procs_sim[0], args.batch_procs[0], args.name

    dir_name = "varying_p/L{}_d{}/".format(L_sim, d_sim)

    file_name = name + '_experiment_L{}_d{}_p{}'.format(L_sim, d_sim,p_gate)

    title = "Evolution for initial position - L={}, # times={}, d={}".format(L_sim, times_sim, d_sim)
    
    simulator = dimers_sim.Simulator(local = False, L=L_sim, times=times_sim, check_interval=check, d=d_sim, prob=p_gate, batch=batch_size, batch_procs_num = batch_procs_num, dir_name="analyses/" + dir_name, file_name=file_name)
  
    results = [simulator.simulate()] 

    #experiment = dimers_analysis.Experiment(file_name,
    #                                 "analyses/" + dir_name,
    #                                  results,
    #                                  description='Varying initial position size experiment for L={}, times={}, d={}, batch_size={}'.format(L_sim, times_sim, d_sim, batch_size))
    
    #experiment.save() 
    dimers_analysis.plot_analyses(results, label = 'd', title=title, name = dir_name + file_name + "_" + 
                             time.strftime("%Y_%m_%d__%H_%M"), save=True)

                         
if __name__ == '__main__':
    print(sys.argv)
    parser = dimers_sim.get_experiment_args()
    args = parser.parse_args()
    print(args)
    Experiments = {'bs' : varying_batch_size, "ic" : varying_initial_conditions, 'pgate' : varying_gate_probabilities, 'q' : quantum}
    dir_names = {'bs' : 'varying_batch_size', "ic" : 'varying_initial_conditions', 'pgate' : 'varying_p', 'q' : 'quantum'}
    try:
        os.mkdir("analyses/{}/L{}_d{}psis/".format(dir_names[args.experiment], args.L[0], args.d[0]))
        os.mkdir("analyses/{}/L{}_d{}/".format(dir_names[args.experiment], args.L[0], args.d[0]))
        
    except:
        print("Folder exists")
    try:
        os.mkdir("figs/{}/L{}_d{}/".format(dir_names[args.experiment], args.L[0], args.d[0]))
    except:
        print("Folder exists")

    experiment_execute = Experiments[args.experiment]
    experiment_execute(args)

