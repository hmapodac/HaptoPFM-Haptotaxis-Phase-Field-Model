#!/bin/tcsh 
module load conda
conda activate /Folder/.conda/envs/phasefield1
python FipyRun2.py
mkdir sweep2
cp ReformatFiles.py sweep2
cp ReformatFilesFracOcc.py sweep2
mv 8/ 9/ 10/ 11/ 12/ 13/ 14/ 15/ sweep2/
cp params2.csv sweep2
cd sweep2
python ReformatFiles.py
python ReformatFilesFracOcc.py
python Analysis_VideoGeneration.py
conda deactivate
