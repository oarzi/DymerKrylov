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
    for arg in args:
        while True:
            name = np.random.randint(1000,10000)
            
            file_name = "temp_sge_files/sge" + str(name) + ".sge"
            with open(file_name, mode="w+", newline=os.linesep) as sge_script:
                arg_with_name = arg + " --name 'cluster{}'".format(name)
                parser = dimers_sim.get_experiment_args()
                arg_parse = parser.parse_args(arg_with_name.split())
                print(arg_parse)
                mem = 1 + (arg_parse.L[0]*(max(arg_parse.times) + max(arg_parse.batch))//1000000000)
                cores = 2 + len(arg_parse.procs_sim) + sum(arg_parse.batch_procs)
                pref = get_prefix(mem ,cores=cores, q='cond-mat-short')
                outs = get_output_files(e='outputs/analysis_error_{}.txt'.format(name),
                                        o='outputs/analysis_output_{}.txt'.format(name))
                multi = get_multi_proc(cores=cores)
                script = get_commands(arg_with_name)

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
    
    b_list = [1e3, 1e4, 1e5, 1e6]
    args_list = ["bs --L 100 --d 60 --times 400 --batch {} --procs_sim 1 --batch_procs 50".format(int(b)) for b in b_list]
    main(args_list)
    
    b_list = [1e3, 1e4, 1e5, 1e6]
    args_list = ["bs --L 500 --d 60 --times 400 --batch {} --procs_sim 1 --batch_procs 50".format(int(b)) for b in b_list]
    main(args_list)
