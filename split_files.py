import dimers_analysis
import os
import numpy as np
from tqdm import tqdm

def split_files(dir_path, before_fac=0.1, after_fac=0.85):
    dir_paths = os.listdir(dir_path)
    for path in tqdm(dir_paths):
        loading = True
        print("Loading: {} ...".format(path))
        while loading:
            try:
                ana = dimers_analysis.Analysis.load(dir_path + "/" + path)
                loading = False
            except:
                continue
        if isinstance(ana, dimers_analysis.Analysis):
            print("Analysis loaded")
            
            #ana_small = dimers_analysis.Analysis(L=ana.L, times=ana.times, d=ana.d, batch=ana.batch,
            #                                     p=ana.p, rho=ana.rho[::6], psis=[],
            #                                     file_name = ana.file_name + "small",
            #                                     dir_name= dir_path + "small/")
            #ana_small.analyze()
            #ana_small.save()
            #del ana_small
            #print("small saved")
            
            #T_before = min(int(before_fac*ana.rho.shape[0]), 3000)
            #T_before = 5000
            #ana_before = dimers_analysis.Analysis(L=ana.L, times=ana.times, d=ana.d, batch=ana.batch,
               #                                   p=ana.p, rho=ana.rho[:T_before,9000:], psis=[],
                #                                  file_name = ana.file_name + "before",
                 #                                 dir_name= dir_path + "before/")
            #ana_before.analyze()
            #ana_before.save()
            #del ana_before
            #print("before saved")

            L_after = min(1 + int(0.3*ana.rho.shape[1]), 200)
            #T_after = int(after_fac*ana.rho.shape[0])
            try:
                ana_after = dimers_analysis.Analysis(L=ana.L, times=ana.times, d=ana.d, batch=ana.batch,
                                                     p=ana.p, rho=ana.rho[40000:, :80], psis=[],
                                                     file_name = ana.file_name + "after",
                                                     dir_name= dir_path + "after/")
                ana_after.analyze()
                ana_after.save()
                del ana_after
                print("after saved")
            except Exception as e:
                print("after failed {}: {}".format(type(e), e))
                pass
                
if __name__ == '__main__':

    dir_path = 'analyses/varying_p/L1200_d1150'
    
    split_files(dir_path)
