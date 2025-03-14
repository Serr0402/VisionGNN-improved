import torch
import torch.nn as nn
import torch.nn.functional as F
from utility2 import SimplePatchifier, TwoLayerNN

class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1):
        super().__init__()
        self.k = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()  
        self.in_layer2 = TwoLayerNN(in_features, in_features * 4)
        self.out_layer2 = TwoLayerNN(in_features, in_features * 4)
        self.droppath2 = nn.Identity()  
        self.multi_head_fc = nn.Conv1d(
            in_features * 2, in_features, 1, 1, groups=head_num
        )

    def forward(self, x):
        B, N, C = x.shape

        sim = x @ x.transpose(-1, -2)
        graph = sim.topk(self.k, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)

        # Aggregation
        neighbor_features = x[
            torch.arange(B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph
        ]
        x = torch.stack(
            [x, (neighbor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1
        )

        # Update
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)

        x = self.droppath1(
            self.out_layer1(F.gelu(x).view(B * N, -1)).view(B, N, -1)
        )
        x = x + shortcut

        x = self.droppath2(
            self.out_layer2(F.gelu(self.in_layer2(x.view(B * N, -1)))).view(B, N, -1)
        ) + x

        return x

class VGNN(nn.Module):
    def __init__(
        self,
        out_feature=320,
        num_ViGBlocks=16,
        num_edges=9,
        head_num=1,
        patch_size=16,
        image_size=224,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = image_size

        # Calcular dinámicamente in_features y num_patches
        in_features = 3 * patch_size * patch_size
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        self.patchifier = SimplePatchifier(patch_size=self.patch_size)
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_features, out_feature // 2),
            nn.BatchNorm1d(out_feature // 2),
            nn.GELU(),
            nn.Linear(out_feature // 2, out_feature // 4),
            nn.BatchNorm1d(out_feature // 4),
            nn.GELU(),
            nn.Linear(out_feature // 4, out_feature // 8),
            nn.BatchNorm1d(out_feature // 8),
            nn.GELU(),
            nn.Linear(out_feature // 8, out_feature // 4),
            nn.BatchNorm1d(out_feature // 4),
            nn.GELU(),
            nn.Linear(out_feature // 4, out_feature // 2),
            nn.BatchNorm1d(out_feature // 2),
            nn.GELU(),
            nn.Linear(out_feature // 2, out_feature),
            nn.BatchNorm1d(out_feature),
        )
        self.pose_embedding = nn.Parameter(torch.rand(num_patches, out_feature))

        self.blocks = nn.Sequential(
            *[
                ViGBlock(out_feature, num_edges, head_num)
                for _ in range(num_ViGBlocks)
            ]
        )

    def forward(self, x):
        x = self.patchifier(x)
        B, N, C, H, W = x.shape
        x = self.patch_embedding(x.view(B * N, -1)).view(B, N, -1)
        x = x + self.pose_embedding

        x = self.blocks(x)

        return x

class Classifier(nn.Module):
    def __init__(
        self,
        out_feature=320,
        num_ViGBlocks=16,
        hidden_layer=1024,
        num_edges=9,
        head_num=1,
        n_classes=10,
        patch_size=16,
        image_size=224,
    ):
        super().__init__()
        self.backbone = VGNN(
            out_feature=out_feature,
            num_ViGBlocks=num_ViGBlocks,
            num_edges=num_edges,
            head_num=head_num,
            patch_size=patch_size,
            image_size=image_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        self.predictor = nn.Sequential(
            nn.Linear(out_feature * num_patches, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Linear(hidden_layer, n_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        B, N, C = features.shape
        x = self.predictor(features.view(B, -1))
        return features, x

