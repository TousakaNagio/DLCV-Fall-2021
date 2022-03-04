filename='model2790.pth'
if [ ! -f filename ]; then
        wget https://www.dropbox.com/s/ct8mc851w99zicv/model2790.pth?dl=0 -O $filename
fi
python3 ./p2_inference.py --val_csv $1 --val_path $2 --out_path $3 --model_path $filename