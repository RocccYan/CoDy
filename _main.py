import os.path  as osp
import os
import time
import random

import torch
import torch.nn.functional as F

from torch_geometric.datasets import DBLP
from torch_geometric.loader import HGTLoader, NeighborLoader
import torch_geometric.transforms as T

from configs import DataArguments, ModelArguments, TrainingArguments, HfArgumentParser
from data.dataloader import build_dataloaders
from data.utils import AverageNeighborFeatures
from model.model import HGT, CitationPredictor
from train.train import train, test, mlse
from utils import *


def main(data_args, model_args, training_args):
    # epochs = training_args['epochs']
    # target = 'paper'
    target = data_args.target
    task = f'{data_args.dataset}' \
        f'@{model_args.model_name}-{model_args.hidden_channels}-{model_args.num_heads}-{model_args.num_layers}' \
        f'@{training_args.seed}@{training_args.num_samples}@{training_args.lr}'
    # define logger
    logger = setup_logger(log_file=f'./log/{task}.log')
    logger.info('Task {} begin...'.format(task))

    # load data
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '../Datasets/DBLP')
    # # We initialize conference node features with a single one-vector as feature:
    # dataset = DBLP(path, transform=T.Constant(node_types='conference'))
    # data = dataset[0]

    # data_path = osp.normpath(osp.join(osp.dirname(osp.realpath(__file__)), '../Datasets/dblp.v13/'))
    data_path = osp.normpath(osp.join(osp.dirname(osp.realpath(__file__)), f'../Datasets/{data_args.dataset}/'))

    # observation_point_train = 2009
    observation_point_train, observation_point_val, observation_point_test = training_args.observation_points.split('=')
    # observation_point_val = training_args.observation_point_val
    # observation_point_test = training_args.observation_point_test

    train_data = torch.load(osp.join(data_path,f'HAG_{observation_point_train}'))
    # val_data = torch.load(osp.join(data_path,f'HAG_{observation_point_val}'))
    # test_data = torch.load(osp.join(data_path,f'HAG_{observation_point_test}'))

    train_data.validate()
    # val_data.validate()
    # test_data.validate()

    # transform by average neighbor features
    transform_author = AverageNeighborFeatures(
        edge_type=('author', 'writes', 'paper'),source_type='author',target_type='paper')
    transform_venue = AverageNeighborFeatures(
        edge_type=('venue', 'publishes', 'paper'),source_type='venue',target_type='paper')

    train_data = transform_venue(transform_author(train_data))
    # val_data = transform_venue(transform_author(val_data))
    # test_data = transform_venue(transform_author(test_data))

    # to undirected by transform of torch_geometric
    train_data = T.ToUndirected()(train_data)
    # val_data = T.ToUndirected()(val_data)
    # test_data = T.ToUndirected()(test_data)

    # logger.info(data)

    # get train, val, test masks
    # load target papers
    papers_target = load_json(osp.join(data_path,f'papers_target.json'))
    
    # randomly select 120,000 papers
    random.seed(training_args.seed)
    papers_target_sampled = random.sample(papers_target,training_args.training_size)

    # filters are according to the corresponding observation point.
    paper_idx_map_train = load_json(osp.join(data_path,f'paper_idx_map_{observation_point_train}.json'))
    # paper_idx_map_val = load_json(osp.join(data_path,f'paper_idx_map_{observation_point_val}.json'))
    # paper_idx_map_test = load_json(osp.join(data_path,f'paper_idx_map_{observation_point_test}.json'))

    # Create a boolean mask of length num_nodes, initialized with False
    train_data[target].mask = torch.zeros(len(paper_idx_map_train), dtype=torch.bool)
    # val_data[target].mask = torch.zeros(len(paper_idx_map_val), dtype=torch.bool)
    # test_data[target].mask = torch.zeros(len(paper_idx_map_test), dtype=torch.bool)

    input_nodes_train = [paper_idx_map_train[k] for k in papers_target_sampled if k in paper_idx_map_train]
    logger.info(len(input_nodes_train))
    # input_nodes_val = [paper_idx_map_val[k] for k in papers_target_sampled]
    # input_nodes_test = [paper_idx_map_test[k] for k in papers_target_sampled]

    train_data[target].mask[input_nodes_train] = True
    # val_data[target].mask[input_nodes_val] = True
    # test_data[target].mask[input_nodes_test] = True

    # build dataloader with HGTLoader
    # num_samples = {
        # ('author', 'writes', 'paper'): [5,5],
        # ('venue', 'publishes', 'paper'): [1,1],
        # ('paper', 'cites', 'paper'): [5,5],
        # Add all other edge types and their corresponding num_samples here
    # }

    # num_samples = [5]
    num_samples = [training_args.num_samples] * model_args.num_layers
    batch_size = training_args.batch_size

    train_loader = HGTLoader(
        train_data,
        num_samples,
        input_nodes=(target, train_data[target].mask),  
        batch_size=batch_size,
        shuffle=True,
    )

    # val_loader = NeighborLoader(
    #     val_data,
    #     num_samples,
    #     input_nodes=(target, val_data[target].mask),
    #     batch_size=batch_size,
    #     shuffle=False,
    # )

    # test_loader = NeighborLoader(
    #     test_data,
    #     num_samples,
    #     input_nodes=(target, test_data[target].mask),
    #     batch_size=batch_size,
    #     shuffle=False,
    # )

    logger.info('get loaders done.')
    # logger.info(next(iter(train_loader)))
    # os._exit(-1)
    
    # load model
    # TODO: layer more than 1 comes to error. key error 'author'
    # model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=2,data=data,target=target)
    model = CitationPredictor(
        hidden_channels=model_args.hidden_channels, 
        num_heads=model_args.num_heads, 
        num_layers=model_args.num_layers,
        data=train_data,
        target=target,
        num_classes=data_args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    train_data, model = train_data.to(device), model.to(device)

    # Initialize lazy modules.
    with torch.no_grad():
        batch = next(iter(train_loader))
        # logger.info(batch.x_dict)
        # logger.info(batch.x_dict['author'])
        # logger.info(batch.edge_index_dict)
        _ = model(batch.x_dict, batch.edge_index_dict)
        # logger.info(out)
    # os._exit(-1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr, weight_decay=training_args.weight_decay)
    loss_fn = mlse
    early_stopping = EarlyStopping(patience=training_args.patience)
    # Continue with your training process
    for epoch in range(0, training_args.num_epochs):
        start_time = time.time()
        loss_reg,loss_cls,acc,f1 = train(train_loader,model=model,loss_fn=loss_fn,optimizer=optimizer,target=target,logger=logger)
        # out = model(data.x_dict, data.edge_index_dict)
        # mask = data[target].train_mask
        # loss = loss_fn(out[mask], data[target].y[mask])
        # loss = loss_fn(out, data[target].y)
        # train_acc = test(data=train_loader,model=model,target=target)
        # val_acc = test(data=val_loader,model=model,target=target)
        epoch_time = time.time() - start_time

        # logger.info(
        #     f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
        #     f'Val: {val_acc:.4f}, Test: {test_acc:.4f}, Time: {epoch_time:.2f} seconds.'
        # )
        logger.info(
            f'Epoch: {epoch:03d}, LossReg: {loss_reg:.4f}, LossCLS: {loss_cls:.4f},'
            f'Acc: {acc:04f}, F1_Macro:{f1:04f}, Time: {epoch_time:.2f} seconds.')
        
        # early stopping
        val_loss = loss_reg + loss_cls
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break


if __name__ == '__main__':
    # set up args parser
    data_args, model_args, training_args = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments)).parse_args_into_dataclasses()
    main(data_args, model_args, training_args)

