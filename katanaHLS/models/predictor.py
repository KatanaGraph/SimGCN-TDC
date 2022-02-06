import torch

class Predictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Predictor, self).__init__()
        
        self.predictor = torch.nn.Sequential(            
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),            
            torch.nn.Linear(hidden_channels, 128),
            # torch.nn.BatchNorm1d((64)),
            torch.nn.ReLU(),            
            torch.nn.Linear(128, out_channels)
        )
    
    
    def reset_parameters(self):
        """Reinitialize model parameters."""
        for n,l in self.predictor.named_children():            
            print(n)
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()

    
    def forward(self, x):
        x = self.predictor(x)
        return x


if __name__ == '__main__':
    pred = Predictor(64, 256, 1)
    pred.reset_parameters()