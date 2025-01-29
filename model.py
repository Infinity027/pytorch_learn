import torch 
from torch import nn

# Define the model
class RegressionModel(nn.Module):
    def __init__(self, layers, input_features):
        super().__init__()
        self.layers = nn.ModuleList()
        for neuron in layers:
            self.layers.append(nn.Linear(input_features, neuron))
            input_features = neuron

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.sigmoid(x)
        return x
    
if __name__=="__main__":
   layers = [2,2,1]
   model = RegressionModel(layers,3)
