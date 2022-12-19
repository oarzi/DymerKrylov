import dimers_sim
import errno
import os
import time
import shutil
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

def get_output(e='analysis_error.txt', o='analysis_output.txt'):
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

def get_script(path):
    script = """echo "Execute job on host $HOSTNAME at $(date)"
python {}
echo "finished job at $(date)
""".format(path)
    return script
    

def main(path_to_file, chdir_path = "", wd_path=''):
    if chdir_path:
        os.chdir(chdir_path)
    try:
        with open(path_to_file, mode="x+", newline=os.linesep) as sge_script:
            pref = get_prefix(wd_path)
            outs = get_output()
            multi = get_multi_proc()
            script = get_script("lalala")
            
            sge_script.write(pref)
            sge_script.write(outs)
            sge_script.write(multi)
            sge_script.write(script)
            
    except FileExistsError:
        print("{} Already exists".format(path_to_file))
        pass
    

    os.system("qsub {}".format(path_to_file))
    
    
    
if __name__ == '__main__':
    main("ofirrrr.sge")

