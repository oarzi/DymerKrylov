import dimers_sim
import errno
import os
import time
import shutil
import numpy as np

def get_prefix(cores=1, q='cond-mat-short', wd_path=''):
    if wd_path:
        wd = 'wd {}'.format(wd_path)
    else:
        wd = 'cwd'
        
    prefix = """#!/bin/bash
#$ -S /bin/bash
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
    output_files = """##$ -o {}
##$ -e {}
""".format(e, o)
    
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
    print("command={}".format(script))    
    script = script + "echo finished job at $(date)."
    return script

def get_sge_scripts(args):
    sge_files = []
    for arg in args:
        while True:
            name = np.random.randint(1000,10000)
            try:
                with open("temp_sge_files/" + str(name) + ".sge", mode="w+", newline=os.linesep) as sge_script:
                    arg_parse = dimers_sim.get_experiment_args()
                    cores = arg_parse.procs_sim * max(arg_parse.batch_procs)
                    pref = get_prefix(cores=cores)
                    outs = get_output_files(e='outputs/analysis_error_{}.txt'.format(name),
                                             o='outputs/analysis_output_{}.txt'.format(name))
                    multi = get_multi_proc(cores=cores)
                    script = get_commands(arg)

                    sge_script.write(pref)
                    sge_script.write(outs)
                    sge_script.write(multi)
                    sge_script.write(script)

                    sge_files.append("temp_sge_files/" + str(name) + ".sge")
                    
                    break
            except FileExistsError:
                print("{} Already exists! Will try again".format(path_to_file))
        
    return sge_files

def main(args_list, chdir_path = "", wd_path=''):
    
    #try:
    #    os.system("mkdir temp_sge_files")
    #except:
        #os.system("rm -r temp_sge_files")    
     #   os.system("mkdir temp_sge_files")

    sge_files = get_sge_scripts(args_list)
    
    #for sge_file in sge_files:
     #  s   os.system("qsub {}".format(sge_files))
        
    #os.system("rm -r temp_sge_files{}")
        
    return

    
if __name__ == '__main__':
    args1 = "bs --L 100 --d 95 --times 100 --batch 100 --procs_sim 1 --batch_procs 1"
    args_list = [args1]
    main(args_list)
