# MNIST
cd ./dataset
python generate_mnist.py iid - - # for iid and unbalanced setting
python generate_mnist.py iid balance - # for iid and balanced setting
python generate_mnist.py noniid - pat # for pathological noniid and unbalanced setting
python generate_mnist.py noniid - dir # for practical noniid and unbalanced setting


# Train

cd ./system
python main.py -data mnist -m cnn -algo FedAvg -gr 100 -did 0 -go cnn # for FedAvg and MNIST