filename='model18400.pth'
if [ ! -f filename ]; then
        wget https://www.dropbox.com/s/0abaq36balgpat7/model18400.pth?dl=0 -O $filename
fi
python3 ./p1/p1_inference.py --img_dir $1 --save_dir $2 --model_path $filename