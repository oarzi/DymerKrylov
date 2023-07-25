import dimers_analysis
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import lzma

if __name__ == '__main__':
    
    # Load date
    ana_path = 'analyses/varying_p/'
    
    exp400_path = 'L400_d300original/'
    #dir_paths = os.listdir(ana_path+exp400_path)
    #for idx, path in enumerate(dir_paths):
    #    with open(ana_path + exp400_path + path, 'rb') as f:
    #        print("Loading {}/{}: {} ...".format(idx + 1, len(dir_paths) ,path))
    #        ana = pickle.load(f)
    #        print(type(ana))
    #        if isinstance(ana, dimers_analysis.Analysis):
    #            print("Analysis loaded")
    #            print(ana.p, ana.rho.shape)
    #            ana_small = dimers_analysis.Analysis(L=ana.L, times=ana.times, d=ana.d, batch=ana.batch,
     #                                   p=ana.p, rho=ana.rho, psis=[], file_name = ana.file_name, 
      #                                  dir_name= ana.dir_name)
       #         ana_small.save()  
    exp400 = dimers_analysis.Experiment.load(dir_path=ana_path + exp400_path, file_name= "exp_" + exp400_path[:-1])
    
    exp800_path = 'L800_d600/'
    exp800 = dimers_analysis.Experiment.load(dir_path=ana_path +  exp800_path, file_name= "exp_" + exp800_path[:-1])
    
    # Velocity
    t_min = 0
    t_max400 = 1200
    v400 = {ana.p : dimers_analysis.extract_velocity(ana ,0, t_max400)[0][0] for ana in exp400.results}
    t_max800 = 3000
    v800 = {ana.p : dimers_analysis.extract_velocity(ana ,0, t_max800)[0][0] for ana in exp800.results}
    
    plist = [ana.p for ana in exp400.results]
    vlist400 = [v400[ana.p] for ana in exp400.results]
    vlist800 = [v800[ana.p] for ana in exp800.results]
    plt.plot(plist, vlist400, label="v(p), L=400")
    plt.plot(plist, vlist800, label="v(p), L=800")
    plt.legend()
    plt.savefig("figs/compare_v800_v400.png", format='png')
    
    # Scaled distributions collapse  
    t = 0.01
    x_max = 800
    x0 = 600
    res_list = exp800.results
    D_list_800 = [dimers_analysis.fit_scaled_dist(ana, v800[ana.p], t*v800[0.14]/v800[ana.p], ana.L, x0)  for i,ana in enumerate(res_list)]
    
    name800 = "L800d600_scaled_all"
    dimers_analysis.plot_dist_scaled_p(res_list, v800 , [t*v800[0.14]/v800[ana.p] for ana in res_list] , x_max, x0,
                                        D_list_800, save=True,name=name800)
                
    name800 = "L800d600_scaled_upto_18"
    res_list = exp800.results[:19]
    dimers_analysis.plot_dist_scaled_p(res_list, v800 , [t*v800[0.14]/v800[ana.p] for ana in res_list] , x_max, x0,
                                        D_list_800, save=True,name=name800)
                
    name800 = "L800d600_scaled_upfrom_18"
    res_list = exp800.results[18:]
    dimers_analysis.plot_dist_scaled_p(res_list, v800 , [t*v800[0.14]/v800[ana.p] for ana in res_list] , x_max, x0, 
                                        D_list_800, save=True,name=name800)
                
    t = 0.01
    x_max = 400
    x0 = 300
    res_list = exp400.results
    D_list_400 = [dimers_analysis.fit_scaled_dist(ana, v400[ana.p], t*v400[0.14]/v400[ana.p], ana.L, x0)  for i,ana in enumerate(res_list)]
    
    name400 = "L400d300_scaled_all"
    dimers_analysis.plot_dist_scaled_p(res_list, v400 , [t*v400[0.14]/v400[ana.p] for ana in res_list] , x_max, x0, 
                                       D_list_400, save=True,name=name400)
    
    name400 = "L400d300_scaled_upto_22"
    res_list = exp400.results[:22]
    dimers_analysis.plot_dist_scaled_p(res_list, v400 , [t*v400[0.14]/v400[ana.p] for ana in res_list] , x_max, x0, 
                                       D_list_400, save=True,name=name400)
    
    # Steady states
                
    times = np.linspace(0.75, 1, 15)
    exp800_steady_state = dimers_analysis.steady_state(exp800.results, times)
    print(exp800_steady_state)
                
    exp400_steady_state = dimers_analysis.steady_state(exp400.results, times)
    print(exp400_steady_state)
                
    with open("analyses/varying_p/steady_state_data.pickle", "wb") as f:
            pickle.dump((exp800_steady_state, exp400_steady_state), f)
                

