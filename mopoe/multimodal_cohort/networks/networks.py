import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, flags, mod_num):
        super(Encoder, self).__init__()
        self.flags = flags
        self.style_dim = flags.style_dim[mod_num]
        hidden_dim = 256
        self.shared_encoder = nn.Sequential(
            nn.Linear(flags.input_dim[mod_num], hidden_dim),
            nn.ReLU(),
            nn.Dropout(flags.dropout_rate),
        )
        self.class_mu = nn.Linear(hidden_dim, flags.class_dim)
        self.class_logvar = nn.Linear(hidden_dim, flags.class_dim)
        # optional style branch
        if flags.factorized_representation:
            self.style_mu = nn.Linear(hidden_dim, self.style_dim)
            self.style_logvar = nn.Linear(hidden_dim, self.style_dim)

    def forward(self, h):
        h = self.shared_encoder(h)
        if self.flags.factorized_representation and self.style_dim > 0:
            return (self.style_mu(h), self.style_logvar(h), self.class_mu(h),
                    self.class_logvar(h))
        else:
            return None, None, self.class_mu(h), self.class_logvar(h)


class Decoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, flags, mod_num):
        super(Decoder, self).__init__()
        self.flags = flags
        self.style_dim = flags.style_dim[mod_num]
        self.decoder = nn.Sequential(
            # nn.Linear(flags.style_dim + flags.class_dim, 256),
            # nn.ReLU(),
            # nn.Dropout(flags.dropout_rate),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(flags.dropout_rate),
            # nn.Linear(256, flags.input_dim[mod_num]),
            nn.Linear(self.style_dim + flags.class_dim,
                      flags.input_dim[mod_num]),
        )
        self.logvar = nn.Parameter(
            data=torch.FloatTensor(
                1, flags.input_dim[mod_num]).fill_(flags.initial_out_logvar),
            requires_grad=flags.learn_output_scale)

    def forward(self, style_latent_space, class_latent_space):
        if (self.flags.factorized_representation and self.style_dim > 0):
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self.decoder(z)
        return x_hat, self.logvar.exp().pow(0.5).to(z.device)
