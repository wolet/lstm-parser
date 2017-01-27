#/usr/bin/env/bash


root=$1
for n in 1244 19916 2489 39832 4979 9958
do
    for f in `find $root/$n -name \*.conll`
    do
	n_observed_conll=`cat $f | awk '{if(NF < 5)print "_"}'| wc -l`
	n_observed_oracl=`echo $f | sed 's/conll/oracle/' | xargs -I '{}' grep "\[ROOT-ROOT\]\[\]" '{}' | wc -l`
	echo $n $f $n_observed_conll $n_observed_oracl
    done
    echo "___________________________________-"
done

