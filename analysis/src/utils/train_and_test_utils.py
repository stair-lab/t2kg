import torch
import pandas as pd
import torch.nn.functional as F
import os
import copy
import json

# Function to save model weights, without saving embeddings during training
def save_checkpoint(model, optimizer, epoch, loss, train_acc, valid_acc, filename):
    """
    Save the model and optimizer states to a file, without saving embeddings during training.
    
    Parameters:
    - model: The model to be saved.
    - optimizer: The optimizer to be saved.
    - epoch: Current epoch number.
    - loss: The current loss.
    - train_acc: Training accuracy.
    - valid_acc: Validation accuracy.
    - filename: Path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
    }
    # Save checkpoint to file
    torch.save(checkpoint, filename)

def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator, save_model_results=False):
    model.eval()
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']


    return train_acc, valid_acc, test_acc

def train_and_evaluate(model, data, train_idx, split_idx, evaluator, args, save_model_results=False, checkpoint_dir="checkpoints"):
    """
    Train and evaluate the model, saving checkpoints and model weights.
    
    Parameters:
    - model: The model to be trained.
    - data: The dataset.
    - train_idx: Indices for training data.
    - split_idx: Dictionary containing indices for 'train', 'valid', and 'test' splits.
    - evaluator: Object for evaluating model performance.
    - args: Dictionary containing training parameters, including 'lr' and 'epochs'.
    - save_model_results: Boolean indicating whether to save model predictions to a file.
    - checkpoint_dir: Directory to store model checkpoints.
    
    Returns:
    - best_model: The model with the highest validation accuracy.
    - metrics: Dictionary containing final training, validation, and test accuracy for the best model.
    """

    # Initialize model parameters and optimizer
    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=args['lr'])
    loss_fn = F.nll_loss

    best_model = None
    best_valid_acc = 0
    best_epoch = 0

    # Store results for possible recovery
    results_log = []

    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, data, train_idx, optimizer, loss_fn)
        train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
        
        # Save training results after each epoch
        results_log.append({
            'epoch': epoch,
            'loss': loss,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'test_acc': test_acc
        })
        
        # Track the best model based on validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch



        # Print progress
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')

    # Final evaluation of the best model
    train_acc, valid_acc, test_acc = test(best_model, data, split_idx, evaluator, save_model_results=save_model_results)

    # Save final embeddings only after the entire training process
    embeddings = best_model(data)  # Extract embeddings from the best model
    embedding_file = os.path.join(checkpoint_dir, "final_embeddings.csv")
    pd.DataFrame(embeddings.cpu().detach().numpy()).to_csv(embedding_file, index=False)

    print(f'Best model: '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')

    # Return the best model and final metrics
    metrics = {
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'test_acc': test_acc
    }
    return best_model, metrics
