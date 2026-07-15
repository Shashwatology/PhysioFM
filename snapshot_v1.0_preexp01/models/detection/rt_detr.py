import torch
import torch.nn as nn
from transformers import RTDetrForObjectDetection, RTDetrConfig

class RTDETRWrapper(nn.Module):
    """
    Wrapper for RT-DETR (Real-Time DEtection TRansformer) for patient 
    and bed localization in the ICU/clinical setting.
    """
    def __init__(self, pretrained=True, num_classes=2): # e.g. Patient, Bed
        super(RTDETRWrapper, self).__init__()
        
        if pretrained:
            # Load a pre-trained RT-DETR model (e.g. from huggingface)
            # In a real scenario, this would be fine-tuned on our ICU dataset
            self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", ignore_mismatched_sizes=True, num_labels=num_classes)
        else:
            config = RTDetrConfig(num_labels=num_classes)
            self.model = RTDetrForObjectDetection(config)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values (Tensor): Image tensor of shape (B, C, H, W)
        Returns:
            dict: Contains 'logits' and 'pred_boxes'
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs

if __name__ == "__main__":
    model = RTDETRWrapper(pretrained=False, num_classes=2)
    dummy_img = torch.randn(1, 3, 640, 640)
    out = model(dummy_img)
    print(f"Logits shape: {out.logits.shape}")
    print(f"Boxes shape: {out.pred_boxes.shape}")
