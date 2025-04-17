import torch.nn as nn
import torch

class SAdapter_module(nn.Module):
    def __init__(self,
                 d_model=None,
                 middle_dim=None,
                 dropout=0.1,
                 adapter_layernorm_option="in",
                 cancate_features=384,
                 ):
        super().__init__()
        middle_features = middle_dim
        self.n_embd = d_model
        self.activation = nn.ReLU()
        # text-down
        self.text_down_proj = nn.Linear(self.n_embd, middle_features)
        # text-up
        self.text_up_proj = nn.Linear(middle_features, cancate_features)

        # image-down
        self.image_down_proj = nn.Linear(self.n_embd, middle_features)
        # image-up
        self.image_up_proj = nn.Linear(middle_features, cancate_features)

        # share-down
        self.share_down_proj = nn.Linear(self.n_embd, middle_features)
        # share-up
        self.share_up_proj = nn.Linear(middle_features, cancate_features)
        
        self.multi_norm = nn.LayerNorm(self.n_embd)
        self.x_norm = nn.LayerNorm(self.n_embd)

        self.dropout = nn.Dropout(p=dropout)

        # ==== Init weights
        self.text_down_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.text_down_proj.bias)
        self.text_up_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.text_up_proj.bias)

        self.image_down_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.image_down_proj.bias)
        self.image_up_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.image_up_proj.bias)

        self.share_down_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.share_down_proj.bias)
        self.share_up_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.share_up_proj.bias)

        

    def forward(self, x, add_residual=True, mode="text"):
        x = self.x_norm(x)
        # shared features
        shared_x = self.share_down_proj(x)
        shared_x = self.activation(shared_x)
        shared_x = self.dropout(shared_x)
        shared_x = self.share_up_proj(shared_x)
        if mode == "text":
            unimodal_x = self.text_down_proj(x)
            unimodal_x = self.activation(unimodal_x)
            unimodal_x = self.dropout(unimodal_x)
            unimodal_x = self.text_up_proj(unimodal_x)

        else:
            unimodal_x = self.image_down_proj(x)
            unimodal_x = self.activation(unimodal_x)
            unimodal_x = self.dropout(unimodal_x)
            unimodal_x = self.image_up_proj(unimodal_x)
        x = torch.cat([unimodal_x, shared_x], dim=2)
        x = self.activation(x)
        return x


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()

        self.down_proj = nn.Linear(c_in, c_in // reduction, bias=True)
        self.non_linear_func = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(c_in // reduction, c_in, bias=True)
        self.drop = nn.Dropout(0.1)
        # init weights
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True):
        x = self.down_proj(x)
        x = self.non_linear_func(x)
        x = self.drop(x)
        x = self.up_proj(x)
        return x
    