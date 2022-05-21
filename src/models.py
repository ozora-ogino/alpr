from typing import List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_a = nn.Conv2d(
            in_channels, int(in_channels / 2), kernel_size=(5, 3), padding=(2, 1)
        )
        self.conv_b = nn.Conv2d(
            in_channels, int(in_channels / 2), kernel_size=(3, 1), padding=(1, 0)
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        a = self.conv_a(x)
        b = self.conv_b(x)
        outputs = [a, b]
        return torch.cat(outputs, 1)


class LPRRPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.inception_block = InceptionBlock(in_channels)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.inception_block(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = nn.Dropout2d(0.5)(x)
        x = F.relu(self.fc7(x))
        return x


class LPRHead(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, n_classes: int = 37) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.brnn = nn.RNN(hidden_size, hidden_size, bidirectional=True)
        self.output = nn.Linear(hidden_size * 2, n_classes)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.convs(x)
        x = self.features_to_sequence(x)
        _, h_n = self.brnn(x)
        h_n = h_n.view(-1, self.hidden_size * 2)
        output = self.output(h_n)
        return output

    def features_to_sequence(self, features):
        b, c, h, w = features.size()
        assert h == 1, "the height of conv must be 1"
        features = features.squeeze(2)
        features = features.permute(2, 0, 1)
        return features


features_pointer = None


def forward_hook(module, inputs, outputs):
    global features_pointer
    features_pointer = outputs.detach().clone()


class LPRNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        backbone = torchvision.models.vgg16(pretrained=True).features
        backbone.out_channels = 512
        anchor_generator = AnchorGenerator(
            sizes=((5, 8, 11, 14, 17, 20),), aspect_ratios=((0.2),)
        )
        rpn_head = LPRRPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"], output_size=(4, 20), sampling_ratio=2
        )

        representation_size = 2048
        self.detect = FasterRCNN(
            backbone=backbone,
            # RPN
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_post_nms_top_n_train=100,
            rpn_post_nms_top_n_test=100,
            box_roi_pool=roi_pooler,
            box_head=TwoMLPHead(
                backbone.out_channels
                * roi_pooler.output_size[0]
                * roi_pooler.output_size[1],
                representation_size,
            ),
            box_predictor=FastRCNNPredictor(representation_size, 2),
            # Keep all proposal.
            box_nms_thresh=1,
            box_score_thresh=0,
        )

        def _hook(model):
            def _func(module, _input, output):
                model.roi_pool_out = output.detach().clone()

            return _func

        self.output = LPRHead(512 * 100, 512)
        self.roi_pool_out = None
        self.detect.roi_heads.box_roi_pool.register_forward_hook(_hook(self))

    def forward(self, x, bbox=None, labels=None):
        if not self.training:
            out, labels = self._predict(x)
            return out, labels
        else:
            loss = self._train_step(x, bbox, labels)
            return loss

    def _predict(self, x):
        self.detect.eval()
        self.output.eval()
        out = self.detect(x)

        # print(self.roi_pool_out.view(len(x), -1, 4, 20).shape)
        # labels = self.output(self.roi_pool_out.view(-1, 512, 4, 20))
        labels = self.output(self.roi_pool_out.view(len(x), -1, 4, 20))
        return out, labels

    def _train_step(self, x, bbox, labels):
        self.detect.train()
        self.output.train()
        # target = dict(image_id=[1 for _ in range(len(bbox))], boxes=bbox)
        # print(target)
        target = [
            {"boxes": b.unsqueeze(0), "labels": torch.ones((1,), dtype=torch.int64)}
            for b in bbox
        ]
        # loss, out = self.detect(x, target)
        losses = self.detect(x, target)
        # print(self.roi_pool_out.shape)
        # pred_labels = self.output(self.roi_pool_out.view(-1, 512, 4, 20))

        # pred_labels = self.output(self.roi_pool_out.view(len(x), -1, 4, 20))
        # print(pred_labels.shape)
        loss = (
            losses["loss_classifier"]
            + losses["loss_box_reg"]
            # + nn.CTCLoss()(pred_labels, labels, len(pred_labels), len(labels))
        )
        return loss


# a = torch.rand(3, 200, 200)
# detect.eval()
# out = detect([a, a])
# print(detect.roi_heads.box_roi_pool)
# print(out[0]["boxes"][0])
# print(out[0])
# print(cls(a.unsqueeze(0)).shape)
# ctc_loss = torch.nn.CTCLoss()

# m = LPRNet()
# m.eval()
# detection, labels = m([a, a])
# print(detection[0]["boxes"].shape)
