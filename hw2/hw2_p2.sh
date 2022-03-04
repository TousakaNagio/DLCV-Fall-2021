filename='26_generate.pt'
if [ ! -f filename ]; then
        wget https://www.dropbox.com/s/7tlpwcvvpyrlay9/26_generator.pt?dl=0 -O $filename
fi
python3 ./p2/p2_generate.py --save_dir $1 --model_path $filename