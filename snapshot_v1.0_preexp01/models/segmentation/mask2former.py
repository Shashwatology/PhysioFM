import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

class Mask2FormerWrapper(nn.Module):
    """
    Wrapper for Mask2Former for precise pixel-wise segmentation of the 
    patient's exposed skin and chest ROI, ignoring blankets/tubing.
    """
    def __init__(self, pretrained=True, num_classes=3): # e.g. Skin, Chest, Background
        super(Mask2FormerWrapper, self).__init__()
        
        if pretrained:
            # We use a base pre-trained model and adjust the classification head
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-tiny-coco-instance",
                ignore_mismatched_sizes=True, 
                num_labels=num_classes
            )
        else:
            config = Mask2FormerConfig(num_labels=num_classes)
            self.model = Mask2FormerForUniversalSegmentation(config)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values (Tensor): Image tensor of shape (B, C, H, W)
        Returns:
            dict: Contains 'class_queries_logits' and 'masks_queries_logits'
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs

if __name__ == "__main__":
    model = Mask2FormerWrapper(pretrained=False, num_classes=3)
    dummy_img = torch.randn(1, 3, 512, 512)
    out = model(dummy_img)
    print(f"Class logits shape: {out.class_queries_logits.shape}")
    print(f"Masks logits shape: {out.masks_queries_logits.shape}")
