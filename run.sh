for slot in {0..10..1}
do
    for shift in {1..6..1}
    do
        echo "Doing slot $slot shift $shift"
        python3 ./main.py --experiment split_cifar100_sc_5 --approach ewc_coscl --lamb 40000 --lamb1 0.02 --use_TG --s_gate 100 --seed 0 --slot $slot --shift $shift
    done
done
