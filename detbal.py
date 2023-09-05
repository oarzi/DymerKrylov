# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:22:53 2023

@author: ofira
"""
import dimers_util
import time

def main():
    
    st = time.process_time()
    states2, state_vars2 = dimers_util.check_detailed_balance3(5, 20000, 4, dimers_util.Gate2, size=1000, interval=100,prob_hop=0.5, save=True)
    et = time.process_time()
    print("check_detailed_balance3 took {}".format(et-st))

    st = time.process_time()
    states2, state_vars2 = dimers_util.check_detailed_balance2(5, 20000, 4, dimers_util.Gate2, size=1000, interval=100,prob_hop=0.5, save=True)
    et = time.process_time()
    print("check_detailed_balance2 took {}".format(et-st))
    
    st = time.process_time()
    states2, state_vars2 = dimers_util.check_detailed_balance(5, 20000, 4, dimers_util.Gate2, size=1000, interval=100,prob_hop=0.5, save=True)
    et = time.process_time()
    print("check_detailed_balance took {}".format(et-st))
    

if __name__ == '__main__':
	main()