import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.lin0 = nn.utils.weight_norm(nn.Linear(latent_size+3, 512))
        self.lin1 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin2 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin3 = nn.utils.weight_norm(nn.Linear(512, 253))
        self.lin4 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin5 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin6 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin7 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin8 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.lin0(x_in)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.lin2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.lin3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = torch.cat((x, x_in), dim=1)

        x = self.lin4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.lin5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.lin6(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.lin7(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.lin8(x)

        return x
