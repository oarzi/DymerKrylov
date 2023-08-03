import dimers_analysis

def split_files(dir_path):
    dir_paths = os.listdir(dir_path)
    for idx, path in enumerate(dir_paths):
        print("Loading {}/{}: {} ...".format(idx + 1, len(dir_paths) ,path))
        ana = dimers_analysis.Analysis.load(dir_path + path)
        if isinstance(ana, dimers_analysis.Analysis):
            print("Analysis loaded")
            ana_small = dimers_analysis.Analysis(L=ana.L, times=ana.times, d=ana.d, batch=ana.batch,
                                        p=ana.p, rho=ana.rho[::6], psis=[], file_name = ana.file_name + "small", 
                                        dir_name= dir_path[:-1] + "small/")
            ana_small.save()
            del ana_small
            where50 = np.argwhere(ana.analysis['Mean'] < 50).T[0]
            if (where50.size > 0):
                ana_before = dimers_analysis.Analysis(L=ana.L, times=ana.times, d=ana.d, batch=ana.batch,
                                            p=ana.p, rho=ana.rho[:where50[0]], psis=[], file_name = ana.file_name + "before", 
                                            dir_name= dir_path[:-1] + "before/")
                ana_before.save()
                del ana_before
                
                L_after = 1 + int(0.2*ana.rho.shape[1])
                ana_after = dimers_analysis.Analysis(L=ana.L, times=ana.times, d=ana.d, batch=ana.batch,
                                            p=ana.p, rho=ana.rho[where50[0]:, :L_after], psis=[], file_name = ana.file_name + "after", 
                                            dir_name= dir_path[:-1] + "after/")
                ana_after.save()
                del ana_after
                
if __name__ == '__main__':
    dir_path400 = 'analyses/varying_p/L400_d300/'
    dir_path800 = 'analyses/varying_p/L800_d600/'

    split_files(dir_path400)
    split_files(dir_path800)