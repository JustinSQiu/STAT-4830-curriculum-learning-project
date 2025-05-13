from transformers import BertConfig, BertForMaskedLM
from torch.optim import Adam
from tqdm import tqdm
from curriculum_learning.dataset import get_dataloader_for_grade

# Initialize model
config = BertConfig(
    num_hidden_layers=12,
    vocab_size=30522,  # Match your tokenizer's vocab size
    hidden_size=768,
    num_attention_heads=12,
)
model = BertForMaskedLM(config)
model.config.update({"architectures": [model.__class__.__name__]})

# Modified parameter grouping function
def get_parameter_groups(model, grade_params, base_lr=1e-4):
    max_layer = grade_params["max_layer"]
    decay = grade_params["decay"]
    param_groups = []
    
    # Access embeddings through bert submodule
    param_groups.append({
        "params": model.bert.embeddings.parameters(),
        "lr": base_lr
    })
    
    # Access encoder layers through bert submodule
    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        if layer_idx <= max_layer:
            lr = base_lr * (decay ** layer_idx)
            layer.requires_grad_(True)
            param_groups.append({
                "params": layer.parameters(),
                "lr": lr
            })
        else:
            layer.requires_grad_(False)
    
    # Add MLM head parameters (instead of classifier)
    param_groups.append({
        "params": model.cls.parameters(),
        "lr": base_lr
    })
    
    return param_groups

# Curriculum setup
curriculum = {
    1: {"max_layer": 5, "decay": 0.1},
    2: {"max_layer": 6, "decay": 0.1},
    # ... continue for grades 3-12
}

# Training loop
base_lr = 1e-4
num_epochs_per_grade = 3

for grade in range(1, 13):  # Assuming grades 1-12
    grade_params = curriculum[grade]
    param_groups = get_parameter_groups(model, grade_params, base_lr)
    # Add this after parameter group setup
    print(f"\nGrade {grade} Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name}")
    optimizer = Adam(param_groups)
    
    train_loader = get_dataloader_for_grade(grade)
    
    model.train()
    for epoch in range(num_epochs_per_grade):
        for batch in tqdm(train_loader, desc=f"Grade {grade}, Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()