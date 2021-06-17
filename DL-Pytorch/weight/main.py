from cnn_weight import Network
from torch.nn.parameter import Parameter

if __name__ == '__main__':
    network = Network()
    print("Network.conv1:")
    print(network.conv1)
    # Accessing The Layer Weights
    print("Accessing Network.conv1")
    print(network.conv1.weight)
    print(network.conv1.weight.shape)
    print("Accessing Network.conv2")
    print(network.conv2.weight.shape)
    print("Accessing Network.out")
    print(network.out.weight.shape)
    # Parameter
    print("Parameter Network.conv1")
    print(network.conv1.parameters)
    print("=========================")
    for param in network.parameters():
        print(param.shape)
    print("=========================")
    for name, param in network.named_parameters():
        print(name, '\t\t', param.shape)
