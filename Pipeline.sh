#!/bin/bash
cd /export/b05/pengyuhe/ML_Project_2/KNN_keep_punc_unstem
source activate ML_Project
python Generate_clean_review.py
python KNN.py
