out_dir="results/temp"

cmd="python3 train.py \
  --out-dir $out_dir\
  --cuda \
"

logfile="$out_dir/train.log"

cmd+=" | tee $logfile"

echo $cmd
eval $cmd
