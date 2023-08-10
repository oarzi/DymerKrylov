import dimers_analysis
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import lzma

def test():
    # Load date
    ana_path = 'analyses/varying_p/'
    
    exp30_path = 'L30_d25'
    exp30 = dimers_analysis.Experiment.load(dir_path=ana_path + exp30_path, file_name= "exp_" + exp30_path[:-1])

    # Velocity
    t_min = 0
    t_max30 = 100
    v30 = {ana.p : dimers_analysis.extract_velocity(ana ,0, t_max30)[0][0] for ana in exp30.results}

    
    plist = [ana.p for ana in exp30.results]
    vlist30 = [v30[ana.p] for ana in exp30.results]
    plt.plot(plist, vlist30, label="v(p), L=30")
    plt.legend()
    plt.savefig("figs/compare_v30.png", format='png')
    
    # Scaled distributions collapse  
    t = 0.01
    x_max = 30
    x0 = 25
    res_list = exp30.results
    D_list_30 = [dimers_analysis.fit_scaled_dist(ana, v30[ana.p], t*v30[0.15]/v30[ana.p], ana.L, x0)  for i,ana in enumerate(res_list)]
    
    name30 = "L30d25_scaled_all"
    dimers_analysis.plot_dist_scaled_p(res_list, v30 , [t*v30[0.15]/v30[ana.p] for ana in res_list] , x_max, x0,
                                        D_list_30, save=True, name=name30)

    # Steady states
                
    times = np.linspace(0.75, 1, 15)
    exp30_steady_state = dimers_analysis.steady_state(exp30.results, times)
    print(exp30_steady_state)

                
    with open("analyses/varying_p/steady_state_data_test.pickle", "wb") as f:
            pickle.dump((exp30_steady_state, ), f)
                
    print("End test")
    return


def compare400800():
    # Load date
    ana_path = 'analyses/varying_p/'
    
    exp400_path = 'L400_d300before/'
    exp400 = dimers_analysis.Experiment.load(dir_path=ana_path + exp400_path)
    
    exp800_path = 'L800_d600before/'
    exp800 = dimers_analysis.Experiment.load(dir_path=ana_path +  exp800_path)
    dimers_analysis.plot_analyses(exp800.results, "p", save=True, title='', name='L800_d600', log_scale_x=False, log_scale_y=False, t_max=12000)
    dimers_analysis.plot_analyses(exp400.results, "p", save=True, title='', name='L400_d300', log_scale_x=False, log_scale_y=False, t_max=12000)
    
    # Velocity
    t_min = 0
    t_max400 = 3000
    v400 = {ana.p : dimers_analysis.extract_velocity(ana ,t_min, t_max400)[0][0] for ana in exp400.results}
    print(v400)
    t_max800 = 6000
    v800 = {ana.p : dimers_analysis.extract_velocity(ana ,t_min, t_max800)[0][0] for ana in exp800.results}
    print(v800)
    
    plt.close('all')
    plist = [ana.p for ana in exp400.results]
    vlist400 = [-v400[ana.p] for ana in exp400.results]
    vlist800 = [-v800[ana.p] for ana in exp800.results]
    plt.plot(plist, vlist400, label="v(p), L=400")
    plt.plot(plist, vlist800, label="v(p), L=800")
    plt.legend()
    plt.savefig("figs/compare_v800_v400.png", format='png')
  
    
    # Scaled distributions collapse  
    t = 0.02
    x_max = 800
    x0 = 600
    res_list = exp800.results
    

    D_list_800 = [dimers_analysis.fit_scaled_dist(ana, v800[ana.p], 3600/ana.rho.shape[0], ana.L, x0)  for i,ana in enumerate(res_list)]

    
    #name800 = "L800d600_scaled_all"
    #dimers_analysis.plot_dist_scaled_p(res_list, v800 , [ min(t*v800[0.84]/v800[ana.p], 0.99) for ana in res_list] , x_max, x0,
       #                                 D_list_800, save=True,name=name800)
       
    # velocity old = min(2*t*v800[0.84]/v800[ana.p], 0.99)
    for i in range(26):
        dimers_analysis.plot_dist_scaled_p([res_list[i]]*10, v800 , [ 500*j/res_list[i].rho.shape[0] for j in range(1,11)] , x_max, x0,
                                       [D_list_800[15]]*10, save=True,name="L800d600_scaled_{}".format(i))
        plt.close()
                                       
                
    name800 = "L800d600_scaled_6_to_all"
    # velocity old = min(2*t*v800[0.84]/v800[ana.p], 0.99)
    dimers_analysis.plot_dist_scaled_p(res_list[6:], v800 , [ 2400/ana.rho.shape[0] for ana in res_list[6:]] , x_max, x0,
                                       D_list_800, save=True,name=name800)
                
    name800 = "L800d600_scaled_0_to_6"
    dimers_analysis.plot_dist_scaled_p(res_list[0:7], v800 , [ 2400/ana.rho.shape[0] for ana in res_list[0:7]] , x_max, x0, 
                                        D_list_800, save=True,name=name800)
                                        
    #for i in range(1,6):
    #        dimers_analysis.plot_dist_scaled_p(exp800.results[:5*i], v800 , [ min(t*v800[0.84]/v800[ana.p], 0.99) for ana in exp800.results[:5*i]] , x_max, x0, 
    #                                    D_list_800, save=True,name="L800d600_scaled_upto_{}".format(5*i))
    
                
    #t = 0.01
    #x_max = 400
    #x0 = 300
    #res_list = exp400.results
    #D_list_400 = [dimers_analysis.fit_scaled_dist(ana, v400[ana.p], t*v400[0.84]/v400[ana.p], ana.L, x0)  for i,ana in enumerate(res_list)]
    
    #name400 = "L400d300_scaled_all"
    #dimers_analysis.plot_dist_scaled_p(res_list, v400 , [t*v400[0.84]/v400[ana.p] for ana in res_list] , x_max, x0, 
    #                                  D_list_400, save=True,name=name400)
    
    #name400 = "L400d300_scaled_upto_22"
    #res_list = exp400.results[:22]
    #dimers_analysis.plot_dist_scaled_p(res_list, v400 , [t*v400[0.84]/v400[ana.p] for ana in res_list] , x_max, x0, 
    #                                   D_list_400, save=True,name=name400)
    
    # Steady states
    
    #exp400_path = 'L400_d300after/'
    #exp400 = dimers_analysis.Experiment.load(dir_path=ana_path + exp400_path)
    
    #exp800_path = 'L800_d600after/'
    #exp800 = dimers_analysis.Experiment.load(dir_path=ana_path +  exp800_path)
                
    #times = np.linspace(0.75, 1, 15)
    #exp800_steady_state = dimers_analysis.steady_state(exp800.results, times)
    #print(exp800_steady_state)
                
    #exp400_steady_state = dimers_analysis.steady_state(exp400.results, times)
    #print(exp400_steady_state)
                
    #with open("analyses/varying_p/steady_state_data.pickle", "wb") as f:
    #       pickle.dump((exp800_steady_state, exp400_steady_state), f)
                


if __name__ == '__main__':
    compare400800()
