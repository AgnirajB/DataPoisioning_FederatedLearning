## declare an array variable
declare -a clients_arr=(10 30 50)
# declare -a algorithms=("FedAvg" "FedProx" "FedProto")
declare -a algorithms=("FedProto")

## now loop through the above array
for clients in ${clients_arr[@]}
do
   for algorithm in ${algorithms[@]}
   do
        echo $clients
        echo $algorithm
        cd ./dataset
        python generate_mnist.py iid balance - $clients
        cd ../system
        python main.py -data mnist -m cnn -algo $algorithm -gr 100 -did 0 -go cnn --num_clients $clients
        cd ../
   done
   # or do whatever with individual element of the array
done
