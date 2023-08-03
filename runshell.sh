#!/bin/bash

#cp analysis_output.txt analysis_output_old.txt
#rm analysis_output.txt

#cp analysis_error.txt analysis_error_old.txt
#rm analysis_error.txt

chmod u+x ./submit_analysis.sge
qsub ./submit_analysis.sge
