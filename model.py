import torch
import torch.nn as nn
import torchvision.models as models

class MultiAttributeModel(nn.Module):
    def __init__(self, backbone_name="mobilenet_v3_small", num_classes_dict=None):
        super().__init__()
        if num_classes_dict is None:
            num_classes_dict = {
                "gender":2,
                "age":5,
                "hair_length":3,
                "upper_clothing":6,
                "lower_clothing":6,
                "accessories":3  # multi-label
            }

        # Backbone
        if backbone_name=="mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights="DEFAULT")
            self.features = backbone.features
            in_features = backbone.classifier[0].in_features
        else:
            raise NotImplementedError

        # Heads
        self.gender_head = nn.Linear(in_features, num_classes_dict["gender"])
        self.age_head = nn.Linear(in_features, num_classes_dict["age"])
        self.hair_head = nn.Linear(in_features, num_classes_dict["hair_length"])
        self.upper_clothing_head = nn.Linear(in_features, num_classes_dict["upper_clothing"])
        self.lower_clothing_head = nn.Linear(in_features, num_classes_dict["lower_clothing"])
        self.accessories_head = nn.Linear(in_features, num_classes_dict["accessories"])

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
        return {
            "gender": self.gender_head(x),
            "age": self.age_head(x),
            "hair_length": self.hair_head(x),
            "upper_clothing": self.upper_clothing_head(x),
            "lower_clothing": self.lower_clothing_head(x),
            "accessories": torch.sigmoid(self.accessories_head(x))  # multi-label
        }
