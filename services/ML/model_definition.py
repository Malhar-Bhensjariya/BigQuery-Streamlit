import torch
import torch.nn as nn

class PyTorchModel(nn.Module):
    """Flexible neural network model that adapts to problem type"""
    def __init__(self, input_size, problem_type, num_classes=None, 
                 hidden_layers=[64, 32], dropout_rate=0.2, l2_reg=0.01):
        super(PyTorchModel, self).__init__()
        self.problem_type = problem_type
        
        # Create hidden layers
        layers = []
        prev_size = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        self.hidden = nn.Sequential(*layers)
        
        # Output layer
        if problem_type == 'binary_classification':
            self.output = nn.Linear(prev_size, 1)
            self.output_act = nn.Sigmoid()
        elif problem_type == 'multiclass_classification':
            self.output = nn.Linear(prev_size, num_classes)
            self.output_act = nn.Softmax(dim=1)
        else:  # regression
            self.output = nn.Linear(prev_size, 1)
            self.output_act = nn.Identity()
        
        # L2 regularization
        self.l2_reg = l2_reg
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return self.output_act(x)
    
    def l2_loss(self):
        """Calculate L2 regularization loss"""
        l2_loss = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            if param.requires_grad:  # Only regularize trainable parameters
                l2_loss = l2_loss + torch.norm(param, 2)
        return self.l2_reg * l2_loss