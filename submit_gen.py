import dimers_sim
import errno
import os
import time
import shutil
import numpy as np

def get_prefix(cores=1, q='cond-mat', wd_path=''):
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

def get_commands(args, file_name):
    script = "echo \"Execute job on host $HOSTNAME at $(date)\"\n"
    #script = script + "chmod u+x {}\n".format(file_name) 
    script = script + "python parallel_analysis.py {}\n".format(args)
    script = script + "echo finished job at $(date)."
    return script

def get_sge_scripts(args):
    sge_files = []
    for arg in args:
        while True:
            name = np.random.randint(1000,10000)
            
            try:
                file_name = "temp_sge_files/sge" + str(name) + ".sge"
                with open(file_name, mode="w+", newline=os.linesep) as sge_script:
                    parser = dimers_sim.get_experiment_args()
                    arg_parse = parser.parse_args(arg.split())
                    cores = 2 + (arg_parse.procs_sim * max(arg_parse.batch_procs))[0]
                    pref = get_prefix(cores=cores)
                    outs = get_output_files(e='outputs/analysis_error_{}.txt'.format(name),
                                            o='outputs/analysis_output_{}.txt'.format(name))
                    multi = get_multi_proc(cores=cores)
                    script = get_commands(arg, file_name)

                    sge_script.write(pref)
                    sge_script.write(outs)
                    sge_script.write(multi)
                    sge_script.write(script)

                    sge_files.append(file_name)
                    
                break
            except FileExistsError:
                print("{} Already exists! Will try again".format(path_to_file))
        
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
    #args_bc = "bs --L 100 --d 95 --times 4000 --batch 100000 400000 1200000 --procs_sim 3 --batch_procs 10 40 50"
    args_ic1 = "ic --L 200 --d 120 --times 3000 --batch 500000 --procs_sim 5 --batch_procs 20"
    args_ic2 = "ic --L 200 --d 100 --times 3000 --batch 500000 --procs_sim 5 --batch_procs 20"
    args_ic3 = "ic --L 200 --d 80 --times 3000 --batch 500000 --procs_sim 5 --batch_procs 20"
    args_ic4 = "ic --L 200 --d 60 --times 3000 --batch 500000 --procs_sim 5 --batch_procs 20"
    args_ic5 = "ic --L 200 --d 40 --times 3000 --batch 500000 --procs_sim 5 --batch_procs 20"
    args_list = [args_ic1, args_ic2, args_ic3, args_ic4, args_ic5]
    main(args_list)
