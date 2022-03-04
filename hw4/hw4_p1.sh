filename='15600-protonet.pth'
if [ ! -f filename ]; then
        wget https://www.dropbox.com/s/0ox6xtck6rjq8rn/15600-protonet.pth?dl=0 -O $filename
fi
python3 ./p1_inference.py --test_csv $1 --test_data_dir $2 --testcase_csv $3 --output_csv $4 --load $filename