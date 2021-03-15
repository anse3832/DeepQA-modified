# FR-IQA condisdering distortion sensitivity 

test code for NTIRE Perceptual Image Quality Assessment (PIQA) Challenge

by just running "test_FR.py", result of inference is saved on "output.txt"

there are four parameters for running "test_FR.py" (from line 132 to 146)
1) GPU_NUM: name of GPU you want to use.
   - ex) GPU_NUM = "0" or GPU_NUM="2"
2) dirname: forder directory where test image exist
3) weights_file: file name of model weights
4) result_score_txt: text file name for storing inference results

if you setting 1), 2), 3) and running the "test_FR.py", inference result is saved in 4)
