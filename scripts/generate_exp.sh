#/usr/bin/env/bash

set -e
parser=$1
exp_root=$2
data_root=$3

layers=2
input_dim=100
hidden_dim=100
epochs=50
lstm_input_dim=100

pretrained_dim=100
action_dim=16
rel_dim=10
pos_dim=12
dev_data=$data_root/raw/val.oracle
test_data=$data_root/raw/tst.oracle
actions_data=$data_root/raw/actions.all
pos_data=$data_root/raw/pos.all
vocabulary_data=$data_root/raw/voc.all
words=$data_root/raw/sskip.100.vectors
other_options="-t --use_pos_tags"

# Data Size
for n in 39832 19916 9958 4979 2489 1244
do
    # RANDOM
     for seed in 0 1 2 3 4 5 6 7 8 9
     do
	 training_data=$data_root/cl/$n/random/$seed/
	 options=$parser
	 curriculum=1
	 output_root=$exp_root/data_size/$n/$curriculum
	 for var in layers input_dim hidden_dim pretrained_dim rel_dim pos_dim lstm_input_dim training_data dev_data test_data actions_data pos_data vocabulary_data epochs words curriculum output_root
	 do
	     options="$options --$var ${!var}"
	 done
	 echo $options $other_options
     done

     # SORTED
     training_data=$data_root/cl/$n/sorted/
     options=$parser
     curriculum=2
     output_root=$exp_root/data_size/$n/$curriculum
     for var in layers input_dim hidden_dim pretrained_dim rel_dim pos_dim lstm_input_dim training_data dev_data test_data actions_data pos_data vocabulary_data epochs words curriculum output_root
     do
	 options="$options --$var ${!var}"
     done
     echo $options $other_options

     # ONEPASS
     training_data=$data_root/cl/$n/onepass/
     options=$parser
     curriculum=3
     output_root=$exp_root/data_size/$n/$curriculum
     for var in layers input_dim hidden_dim pretrained_dim rel_dim pos_dim lstm_input_dim training_data dev_data test_data actions_data pos_data vocabulary_data epochs words curriculum output_root
     do
	 options="$options --$var ${!var}"
     done
     echo $options $other_options

     #BABYSTEP
     training_data=$data_root/cl/$n/babystep/
     options=$parser
     curriculum=4
     output_root=$exp_root/data_size/$n/$curriculum
     for var in layers input_dim hidden_dim pretrained_dim rel_dim pos_dim lstm_input_dim training_data dev_data test_data actions_data pos_data vocabulary_data epochs words curriculum output_root
     do
	 options="$options --$var ${!var}"
     done
     echo $options $other_options
 done

#MODEL SIZE
n=39832

for layers in 2 1
do
    for dim in 128 64 32 16 8 4 2
    do
	input_dim=$dim
	hidden_dim=$dim
	lstm_input_dim=$dim

	# RANDOM
	for seed in 0 1 2 3 4 5 6 7 8 9
	do
	    training_data=$data_root/cl/$n/random/$seed/
	    options=$parser
	    curriculum=1
	    output_root=$exp_root/model_size/$n/$curriculum
	    for var in layers input_dim hidden_dim pretrained_dim rel_dim pos_dim lstm_input_dim training_data dev_data test_data actions_data pos_data vocabulary_data epochs words curriculum output_root
	    do
		options="$options --$var ${!var}"
	    done
	    echo $options $other_options
	done
	# SORTED
	training_data=$data_root/cl/$n/sorted/
	options=$parser
	curriculum=2
	output_root=$exp_root/model_size/$n/$curriculum
	for var in layers input_dim hidden_dim pretrained_dim rel_dim pos_dim lstm_input_dim training_data dev_data test_data actions_data pos_data vocabulary_data epochs words curriculum output_root
	do
	    options="$options --$var ${!var}"
	done
	echo $options $other_options

	# ONEPASS
	training_data=$data_root/cl/$n/onepass/
	options=$parser
	curriculum=3
	output_root=$exp_root/model_size/$n/$curriculum
	for var in layers input_dim hidden_dim pretrained_dim rel_dim pos_dim lstm_input_dim training_data dev_data test_data actions_data pos_data vocabulary_data epochs words curriculum output_root
	do
	    options="$options --$var ${!var}"
	done
	echo $options $other_options

	#BABYSTEP
	training_data=$data_root/cl/$n/babystep/
	options=$parser
	curriculum=4
	output_root=$exp_root/model_size/$n/$curriculum
	for var in layers input_dim hidden_dim pretrained_dim rel_dim pos_dim lstm_input_dim training_data dev_data test_data actions_data pos_data vocabulary_data epochs words curriculum output_root
	do
	    options="$options --$var ${!var}"
	done
	echo $options $other_options
    done
done
