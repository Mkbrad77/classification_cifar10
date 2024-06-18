Getting started
===============

This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.
cd /home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python cifar10_classification/modeling/grid_search.py