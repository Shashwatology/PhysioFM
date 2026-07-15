import torch
import torch.nn as nn

class ClinicalReasoningEngine(nn.Module):
    """
    The Clinical Reasoning Engine.
    Maps the fused Physiological Representation into clinical risk scores 
    while maintaining interpretable logic paths (e.g., separating 
    respiratory failure paths from cardiac failure paths).
    """
    def __init__(self, embed_dim=512, num_risk_classes=1):
        super(ClinicalReasoningEngine, self).__init__()
        
        # We split the latent representation into "logical" branches 
        # to force the model to reason about specific subsystems.
        self.branch_dim = embed_dim // 2
        
        # Branch 1: Cardiopulmonary Logic
        self.cardio_logic = nn.Sequential(
            nn.Linear(self.branch_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Branch 2: Neuromuscular/Activity Logic (Motion, Posture, Falls)
        self.activity_logic = nn.Sequential(
            nn.Linear(self.branch_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Final Risk Aggregation
        # Outputs a continuous risk score (e.g. probability of deterioration)
        self.risk_head = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_risk_classes)
        )

    def forward(self, physio_rep):
        """
        Args:
            physio_rep (Tensor): Fused physiological representation (B, embed_dim)
        Returns:
            Tensor: Risk score logits (B, num_risk_classes)
            dict: Intermediate logic states for explainability (SHAP)
        """
        # Split the representation
        cardio_in, activity_in = torch.split(physio_rep, self.branch_dim, dim=-1)
        
        # Process logical branches
        cardio_state = self.cardio_logic(cardio_in)
        activity_state = self.activity_logic(activity_in)
        
        # Aggregate
        combined_logic = torch.cat([cardio_state, activity_state], dim=-1)
        risk_score = self.risk_head(combined_logic)
        
        # Return intermediate states for XAI hooking
        xai_states = {
            "cardio_state": cardio_state,
            "activity_state": activity_state
        }
        
        return risk_score, xai_states

if __name__ == "__main__":
    model = ClinicalReasoningEngine(embed_dim=512)
    dummy_physio = torch.randn(4, 512) # Batch 4
    score, states = model(dummy_physio)
    print(f"Risk Score shape: {score.shape}") # Expected: (4, 1)
    print(f"Cardio state shape: {states['cardio_state'].shape}")
