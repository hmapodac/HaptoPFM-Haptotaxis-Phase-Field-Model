#!/bin/tcsh 
module load conda
conda activate /Folder/.conda/envs/phasefield1
python FipyRun1.py
mkdir sweep1
cp ReformatFiles.py sweep1
cp ReformatFilesFracOcc.py sweep1
mv 0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ sweep1/
cp params1.csv sweep1
cd sweep1
python ReformatFiles.py
python ReformatFilesFracOcc.py
python Analysis_VideoGeneration.py
conda deactivate
