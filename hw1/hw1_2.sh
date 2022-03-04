filename='model.pkl-12201'
if [ ! -f $filename ]; then
	wget https://www.dropbox.com/s/y3g57t32bc1zaf7/model.pkl-12201?dl=0 -O $filename
fi
python3 ./p2/inference.py --img_dir $1 --save_dir $2 --model_path $filename

#collaborator thanks to b08902134 曾楊哲