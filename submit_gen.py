import dimers_sim
import errno
import os
import time
import shutil
import numpy as np

def get_prefix(mem, cores=1, q='cond-mat', wd_path=''):
    if wd_path:
        wd = 'wd {}'.format(wd_path)
    else:
        wd = 'cwd'
        
    prefix = """#!/bin/bash\n
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
    output_files = """#$ -o {}
#$ -e {}
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
    script = script + "echo finished job at $(date)."
    return script

def get_sge_scripts(args):
    sge_files = []
    parser = dimers_sim.get_experiment_args()
    for arg in args:
        arg_parse = parser.parse_args(arg.split())
        mem = 1 + (arg_parse.L[0]*(max(arg_parse.times) + max(arg_parse.batch))//1000000000)
        cores = 2 + sum([p*b for p,b in zip(arg_parse.procs_sim, arg_parse.batch_procs)])
        while True:
            name = np.random.randint(1000,10000)
            
            file_name = "temp_sge_files/sge" + str(name) + ".sge"
            with open(file_name, mode="w+", newline=os.linesep) as sge_script:
                
                name = "'cluster{}'".format(name)
                if arg_parse.name[0] != 'def':
                    arg = arg.replace(arg_parse.name[0], arg_parse.name[0] + "_" + name)
                else:
                    arg = arg + " --name " + name

                print(parser.parse_args(arg.split()))

                
                pref = get_prefix(mem ,cores=cores, q='cond-mat-shortx  ')
                outs = get_output_files(e='outputs/analysis_error_{}.txt'.format(name),
                                        o='outputs/analysis_output_{}.txt'.format(name))
                multi = get_multi_proc(cores=cores)
                script = get_commands(arg)

                sge_script.write(pref)
                sge_script.write(outs)
                sge_script.write(multi)
                sge_script.write(script)

                sge_files.append(file_name)
                
            break

        
    return sge_files

def main(args_list, chdir_path = "", wd_path=''):
    
    #try:
    #   os.system("rm -r temp_sge_files")
    #except:
    #   pass    
    #os.system("mkdir temp_sge_files")

    sge_files = get_sge_scripts(args_list)

    for sge_file in sge_files:
        print(sge_file)
        os.system("chmod u+x {}".format(sge_file))
        os.system("qsub {}".format(sge_file))
        
        
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
        args_list = ["pgate --L 400 --d 30 --times 800 --batch 10000 --p {} --procs_sim 1 --batch_procs 80".format(_p) for _p in p_list]

    """
 
    p_list = [0.95, 0.75, 0.5, 0.35, 0.2, 0.15]
    args_list = ["pgate --L 800 --d 60 --times 6000 --batch 1000 --p {} --procs_sim 1 --batch_procs 15 --name permutation".format(_p) for _p in p_list]
    main(args_list)
    
    p_list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    args_list = ["pgate --L 800 --d 60 --times 6000 --batch 5000 --p {} --procs_sim 1 --batch_procs 15 --name permutation".format(_p) for _p in p_list]
    main(args_list)
    
