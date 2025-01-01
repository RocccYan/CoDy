import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score

from torch_geometric.loader import HGTLoader, NeighborSampler
from torch_geometric.data import Data, DataLoader

"""In this training setting, train val and test set are from different graphs, so we 
dont need different mask.
"""

def train(data_input,encoder=None,model=None,loss_fn=None,optimizer=None,target=None,logger=None):
    """
    Notes:
    - data and mode should be defined and as parameters in the function.
    """
    encoder.train()
    model.train()
    total_loss_reg, total_loss_cls = 0, 0
    # with tqdm(total=len(data_input), desc=f'[Training]', unit='batch') as pbar:
    
    # Instantiate accuracy and F1 metric objects for multi-class classification
    predictions = []
    targets = []

    for batch_idx, data in enumerate(data_input):
        optimizer.zero_grad()
        # out_reg, out_cls = model(encoder(data.x_dict, data.edge_index_dict)[target])
        out_reg, out_cls = model(encoder(data.x_dict, data.edge_index_dict, data.edge_time_dict)[target])
        # mask = data[target].train_mask
        # loss = loss_fn(out[mask], data[target].y[mask])
        out_reg = out_reg.squeeze()
        loss_reg = loss_fn(out_reg, data[target].y)

        y_cls = torch.tensor([get_label_reg2cls(y) for y in data[target].y],
                             dtype=torch.long,device=out_cls.device)
        loss_cls = F.cross_entropy(out_cls, y_cls)
        # loss = loss_reg + loss_cls
        loss = loss_cls
        # if batch_idx > 30: os._exit(-1)
        loss.backward()
        optimizer.step()

        # if logger:
        #     logger.info(f"batch_idx: {batch_idx}, Avg Reg Loss:  {float(loss_reg)/out_reg.shape[0]}")
        #     logger.info(f"batch_idx: {batch_idx}, Avg CLS Loss:  {float(loss_cls)/out_cls.shape[0]}")
        
        total_loss_reg += loss_reg.item()
        total_loss_cls += loss_cls.item()
        predictions += out_cls.argmax(dim=-1).tolist()
        targets += y_cls.tolist()

    # eval classification results
    accuracy = accuracy_score(predictions, targets)
    f1_macro = f1_score(predictions, targets, average='macro')
    # if logger:
    #     logger.info(f"Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
    return total_loss_reg, total_loss_cls, accuracy, f1_macro


@torch.no_grad()
def test(dataloader,model=None,target='author'):
    model.eval()
    # TODO: check whether pred is compatible with the target.
    for step, batch in tqdm(enumerate(dataloader)):
    
        out = model(batch.x_dict, batch.edge_index_dict)
        mask = batch.target.val_mask
        acc = (out[mask] == batch[target].y[mask]).sum() / mask.sum()
    # pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []

# Define the evaluation function for regression
@torch.no_grad()
def eval(dataloader,model=None,target='author'):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    labels = []
    for data in dataloader:  # Iterate over each batch in the DataLoader
        out = model(data.x_dict, data.edge_index_dict)
        val_mask = data[target].val_mask
        predictions.append(out[data[val_mask]].detach())
        labels.append(data.y[data[val_mask]].detach())
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return mlse(labels.cpu(), predictions.cpu())  # RMSE


# def mlse(y_hat,y):
#     """
#     mean of log squared error
#     """
#     loss = F.mse_loss(
#         torch.log(torch.max(y_hat,torch.zeros_like(y_hat)) + 0.01),
#         torch.log(y + 0.01)
#     )
#     return loss

def mlse(y_hat,y):
    """
    mean of log squared error
    """
    a = torch.max(y_hat,torch.zeros_like(y_hat))
    b = torch.log(y + 1)
    loss = F.mse_loss(a,b)
    # print(a,'\n',b,'\n',loss)
    return loss

def get_label_reg2cls(y):
    if y < 10:
        return 0
    elif y < 100:
        return 1
    else:
        return 2

