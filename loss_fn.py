import torch
import torch.nn.functional as F

def compute_loss(logits, gold_labels, silver_labels):
    """
    Compute the custom loss function combining gold and silver label losses.
    
    :param logits: Tensor of shape (batch_size, num_sentences, num_classes)
    :param gold_labels: Tensor of shape (batch_size, num_classes)
    :param silver_labels: Tensor of shape (batch_size, num_sentences)
    :return: Combined loss
    """
    batch_size, num_sentences, num_classes = logits.size()
    
    # Compute max logits across the two classes (positive/negative)
    max_logits = logits.max(dim=2)[0]
    
    # Compute softmax weights
    weights = F.softmax(max_logits, dim=1)
    
    # Compute the gold label loss (weighted cross-entropy)
    gold_loss = 0
    for i in range(batch_size):
        for j in range(num_sentences):
            gold_loss += weights[i, j] * F.cross_entropy(logits[i, j].unsqueeze(0), gold_labels[i].unsqueeze(0), reduction='none')
    gold_loss = gold_loss.sum()
    
    # Compute the silver label loss (cross-entropy)
    silver_loss = F.cross_entropy(logits.view(-1, num_classes), silver_labels.view(-1), reduction='none')
    silver_loss = silver_loss.sum()
    
    # Total loss
    total_loss = gold_loss + silver_loss
    return total_loss

# Example usage:
batch_size = 4
num_sentences = 10
num_classes = 2

logits = torch.randn(batch_size, num_sentences, num_classes)
gold_labels = torch.randint(0, num_classes, (batch_size,))
silver_labels = torch.randint(0, num_classes, (batch_size, num_sentences))

loss = compute_loss(logits, gold_labels, silver_labels)
print(f'Loss: {loss.item()}')
