
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_size = 64 * 1
        self.num_layers = 1
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.InstanceNorm1d(128),
            nn.Dropout(0.),
            # nn.BatchNorm1d(128),
            nn.Linear(128, self.hidden_size),
            nn.ReLU(),
            nn.InstanceNorm1d(self.hidden_size),
            nn.Dropout(0.),
            # nn.BatchNorm1d(self.hidden_size),
        )
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.InstanceNorm1d(self.hidden_size//2),
            nn.Dropout(0.),
            nn.Linear(self.hidden_size//2, num_classes),
        )


    def forward(self, x, seq_len_vec):
        # print(x.size())
        x = self.mlp(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        batch_indices = torch.arange(x.size(0)).to(x.device)
        out = out[batch_indices, seq_len_vec, :]
        out = self.fc(out)
        return out

class GRUClassifier_surrogate(nn.Module):
    def __init__(self, input_size, num_classes, feat_len, num_layer):
        super(GRUClassifier_surrogate, self).__init__()
        self.hidden_size = feat_len
        self.num_layers = num_layer
        self.dropout = 0.3
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.InstanceNorm1d(128),
            nn.Dropout(self.dropout),
            # nn.BatchNorm1d(128),
            nn.Linear(128, self.hidden_size),
            nn.ReLU(),
            nn.InstanceNorm1d(self.hidden_size),
            nn.Dropout(self.dropout),
            # nn.BatchNorm1d(self.hidden_size),
        )
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.InstanceNorm1d(self.hidden_size//2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//2, num_classes),
        )

    def forward(self, x, seq_len_vec):
        # print(x.size())
        x = self.mlp(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        batch_indices = torch.arange(x.size(0)).to(x.device)
        out = out[batch_indices, seq_len_vec, :]
        out = self.fc(out)
        return out
    
class GRUClassifier_sp(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GRUClassifier_sp, self).__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.InstanceNorm1d(128),
            nn.Dropout(0.1),
            # nn.BatchNorm1d(128),
            nn.Linear(128, self.hidden_size),
            nn.ReLU(),
            nn.InstanceNorm1d(self.hidden_size),
            nn.Dropout(0.1),
            # nn.BatchNorm1d(self.hidden_size),
        )
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)

        self.fc_1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.InstanceNorm1d(self.hidden_size//2),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size//2, 5),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.InstanceNorm1d(self.hidden_size//2),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size//2, 3),
        )

    def forward(self, x):
        # print(x.size())
        x = self.mlp(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out_1 = self.fc_1(out[:, -1, :])
        out_2 = self.fc_2(out[:, -1, :])
        return out_1, out_2