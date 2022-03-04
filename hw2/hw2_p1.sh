wget https://www.dropbox.com/s/46491ve6qo936pg/models.zip?dl=0 -O models.zip
pip install stylegan2-pytorch
unzip models.zip
python3 p1_generate.py --output $1 --load_from 66