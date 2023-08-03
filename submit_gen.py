import dimers_sim
import errno
import os
import time
import shutil
import numpy as np
from itertools import product

def get_prefix(mem, cores=1, q='cond-mat', wd_path=''):
    if wd_path:
        wd = 'wd {}'.format(wd_path)
    else:
        wd = 'cwd'
        
    prefix = """#!/bin/bash\n
#$ -S /bin/bash
#$ -l h_rss=3g
#$ -{} 
#$ -M ofir.arzi@tum.de
#$ -m ea  
    
#$ -pe smp {}
#$ -R y
#$ -b y
#$ -q {}
""".format(wd, cores, q)
    return prefix

def get_output_files(e='analysis_error.txt', o='analysis_output.txt'):
    output_files = """#$ -o {}
#$ -e {}
""".format(o, e)
    
    return output_files


def get_multi_proc(cores=1):
    multi = """## export PATH=\"/mount/packs/intelpython36/bin:$PATH\"
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS={0}  
export OMP_NUM_THREADS={0}
""".format(cores)
    
    return multi

def get_commands(args):
    script = "echo \"Execute job on host $HOSTNAME at $(date)\"\n"
    script = script + "python parallel_analysis.py {}\n".format(args)
    script = script + "echo finished job at $(date)."
    return script

def get_sge_scripts(args):
    sge_files = []
    parser = dimers_sim.get_experiment_args()
    for arg in args:
        arg_parse = parser.parse_args(arg.split())
        print(arg_parse)
        mem = 1 + (arg_parse.L[0]*(max(arg_parse.times) + max(arg_parse.batch))//1000000000)
        cores = 1 + sum([p*b for p,b in zip(arg_parse.procs_sim, arg_parse.batch_procs)])
        while True:
            name = "sge" + str(np.random.randint(1000,10000))
            dir_name = "temp_sge_files/{}".format(name)
            try:
                os.system("mkdir {}".format(dir_name))
            except:
                os.system("rm -r {}/*".format(name))

            file_name = "{}/{}.sge".format(dir_name, arg_parse.experiment + name)
            with open(file_name, mode="w+", newline=os.linesep) as sge_script:
                arg = arg.replace(arg_parse.name, arg_parse.name + "_" + name)
                
                
                pref = get_prefix(mem ,cores=cores, q='cond-mat')
                outs = get_output_files(e='temp_sge_files/{}/error_{}.txt'.format(name, name[3:]),
                                        o='temp_sge_files/{}/output_{}.txt'.format(name, name[3:]))
                multi = get_multi_proc(cores=cores)
                command = get_commands(arg)

                sge_script.write(pref)
                sge_script.write(outs)
                sge_script.write(multi)
                sge_script.write(command)

                sge_files.append((file_name, arg))
                
            break

        
    return sge_files

def main(args_list, chdir_path = "", wd_path=''):
    
    #try:
    #   os.system("rm -r temp_sge_files")
    #except:
    #   pass    
    #os.system("mkdir temp_sge_files")

    sge_files = get_sge_scripts(args_list)

    for job, sge_file in enumerate(sge_files):
        print("Job {}/{}: ".format(job + 1, len(sge_files)) + sge_file[0] + "; " + sge_file[1])
        os.system("chmod u+x {}".format(sge_file[0]))
        os.system("qsub {}".format(sge_file[0]))
        
        
    return

    
if __name__ == '__main__':
    """
    1. Initial conditions example:

        L_list = [300, 600, 1200, 2400]
        args_list = ["ic --L {} --d 60 --times 800 --batch 10000 --procs_sim 1 --batch_procs 40".format(_L) for _L in L_list]
        
    2. Batch size example:

        args_list = ["bs --L 100 --d 60 --times 800 --batch 1000 10000 100000 1000000 --procs_sim 1 --batch_procs 40"]
        
    3. Probability example:
        
        p_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        args_list = ["pgate --L 400 --d 30 --times 800 --check {}--batch 10000 --p {} --procs_sim 1 --batch_procs 80".format(_p) for _p in p_list]
        
    4. Quantum:
        times = [50, 200, 1000, 2000, 4000]
        args_list = ["q --L 34 --d 26 --times {} --check{}".format(t) for t in times]        

    """
    
    #p_list3 = [0.01, 0.03, 0.06, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99]
    #args_list3 = ["pgate --L 100 --d 85 --times 500 --check 100 --batch 1000 --p {} --procs_sim 1 --batch_procs 12".format(_p) for _p in p_list3]
    #main(args_list3)
    
    #p_list1 = [0, 0.0001, 0.001, 0.01, 0.03, 0.06, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99]
    #p_list1 = [0.3]
    #args_list1 = ["pgate --L 400 --d 300 --times 5000 --check 100 --batch 1500 --p {} --procs_sim 1 --batch_procs 24".format(_p) for _p in p_list1]
    #args_list1 = args_list1 + ["pgate --L 400 --d 300 --times 5000 --check 150 --batch 1500 --p {} --procs_sim 1 --batch_procs 24".format(_p) for _p in p_list1[:3]]
    #args_list1 = args_list1 + ["pgate --L 400 --d 300 --times 5000 --check 200 --batch 1500 --p {} --procs_sim 1 --batch_procs 24".format(_p) for _p in p_list1[25:]]
    
    #main(args_list1)
    

    #p_list2 = [0, 0.0001, 0.001, 0.01, 0.03, 0.06, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99]
    #args_list2 = ["pgate --L 800 --d 600 --times 5000 --check 100 --batch 1500 --p {} --procs_sim 1 --batch_procs 36".format(_p) for _p in p_list2[3:25]]
    #args_list2 = args_list2 + ["pgate --L 800 --d 600 --times 5000 --check 150 --batch 2000 --p {} --procs_sim 1 --batch_procs 36".format(_p) for _p in p_list2[:3]]
    #args_list2 = args_list2 + ["pgate --L 800 --d 600 --times 5000 --check 300 --batch 2000 --p {} --procs_sim 1 --batch_procs 36".format(_p) for _p in p_list2[25:]]
    #main(args_list2)

    #p_list3 = [0, 0.0001, 0.001, 0.01, 0.03, 0.06, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99]
    #p_list3 = [0.3, 0.0001, 0.03, 0.93, 0.8, 0.9]    
    args_list3 = ["q --L 30 --d 26 --times 100 --check 100 --p {}".format(_p) for _p in [0]] 
    main(args_list3)
    
