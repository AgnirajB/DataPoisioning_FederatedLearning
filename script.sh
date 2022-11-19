## declare an array variable
declare -a clients_arr=(10 30 50)
declare -a algorithms=("FedAvg" "FedProx" "FedProto")
# declare -a mal_node_perc=(10 20 30)
# declare -a mal_data_perc=(10 20 30)
# declare -a mal_data_perc=(0.1)
# declare -a mal_node_perc=(0 10 20 30)
declare -a mal_node_perc=(20 30)
declare -a mal_data_perc=(1 5 10 15)

## now loop through the above array
for clients in ${clients_arr[@]}
do
   for mp in ${mal_node_perc[@]}
   do
      for md in ${mal_data_perc[@]}
      do
         for algorithm in ${algorithms[@]}
         do
            echo $clients
            echo $algorithm
            /home/rmekala/anaconda3/envs/snoopy/bin/python -u generate_mnist.py iid balance - $clients $mp $md
            cd ./system
            /home/rmekala/anaconda3/envs/snoopy/bin/python -u main.py -data mnist -m cnn -algo $algorithm -gr 100 -did 0 -go cnn --num_clients $clients -mp $mp -md $md
            cd ../
         done

      done

   done
done
