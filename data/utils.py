import json
import os
import re
from tqdm import tqdm 

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, Dataset
import torch
import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from transformers import BertTokenizer, AutoTokenizer
from transformers import BertModel, AutoModel


MAX_LEN = 512
MODEL_PATH_BERT = '../../models/bert-base-uncased'
MODEL_PATH_SCIBERT = '../../models/scibert_scivocab_uncased'


class AverageNeighborFeatures(BaseTransform):
    def __init__(self, edge_type, source_type, target_type, feature_key='x'):
        # edge_type is a tuple (source_node_type, relation_type, target_node_type)
        # source_type is the node type for which features need to be averaged
        # target_type is the node type from which features are averaged
        # feature_key is the key used to access the feature matrix in HeteroData
        self.edge_type = edge_type
        self.source_type = source_type
        self.target_type = target_type
        self.feature_key = feature_key
    def __call__(self, data):
        edge_index = data[self.edge_type].edge_index
        target_features = data[self.target_type][self.feature_key]
        num_sources = data[self.source_type].num_nodes

        # Initialize the source features as zeros
        source_features = torch.zeros((num_sources, target_features.size(1)))

        # Sum the target features for each source node
        for source_id, target_id in edge_index.t().tolist():
            source_features[source_id] += target_features[target_id]

        # Count the number of connections for each source node
        connection_counts = torch.zeros(num_sources)
        for source_id in edge_index[0].tolist():
            connection_counts[source_id] += 1

        # Avoid division by zero for nodes with no connections
        connection_counts = connection_counts.masked_fill(connection_counts == 0, 1)

        # Compute the average by dividing the summed features by the connection counts
        source_features = source_features / connection_counts.unsqueeze(1)

        # Add the averaged features to the source node type in data
        data[self.source_type][self.feature_key] = source_features

        return data


def check_heterograph(data):
    # data.valiadate()
    # Check the number of nodes for each type
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        print(f"Node type '{node_type}' has {num_nodes} nodes.")
    # Check edge indices for each edge type
    for edge_type in data.edge_types:
        print("Edge Type:", edge_type)
        # Unpack the edge_type tuple
        src_type, rel_type, dst_type = edge_type
        # Retrieve edge index tensor for the current edge type
        edge_index = data[src_type, rel_type, dst_type].edge_index
        # print("Edge Index:", edge_index)
        if edge_index.size()[-1] == 0:
            continue
        src_min_index = edge_index[0].min().item()
        src_max_index = edge_index[0].max().item()
        print(f"\tType src {src_type} index interval:({src_min_index},{src_max_index})")
        dst_min_index = edge_index[1].min().item()
        dst_max_index = edge_index[1].max().item()
        print(f"\tType dst {dst_type} index interval:({dst_min_index},{dst_max_index})")


# Helper function to convert edge list to tensor
def edge_list_to_tensor(edge_list, idx_map_src, idx_map_dst):
    if edge_list:
        return torch.tensor(
            [(idx_map_src[src], idx_map_dst[dst]) for src, dst in edge_list],
            dtype=torch.long
        ).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)


def get_citation_5(paper_citation, observation_point):
    """
    Here C_5 means the accumulated citation after another 5 years from
    observation point.
    """
    citation_5 = 0
    for year in range(observation_point+1, observation_point+6):
        citation_5 += paper_citation.get(str(year), 0)
    return citation_5


class TokenizedDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = text_preprocessing(self.data[idx])
        encoded_sent = self.tokenizer.encode_plus(
            text=sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded_sent['input_ids'].flatten(),
            'attention_mask': encoded_sent['attention_mask'].flatten()
        }


def default_json(t):
    return f'{t}'

def load_json(filepath):
    with open(filepath,'r', encoding='utf-8') as f:
        return json.load(f)

def save_as_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=default_json)
    print(f"{filename} saved.")

def get_demo_data(big_data_path, chunk_size=4):
    with open(big_data_path, 'r') as fh:
        demo_data = next(read_by_chunks(fh, chunk_size=chunk_size))
    
    with open(big_data_path+'-demo', 'w') as fh:
        fh.write(demo_data)
    print('demo data.')

def read_by_chunks(file_obj, chunk_size=4, unit='k'):
    """
    Lazy function to read a file piece by piece.
    Default chunk size: 4kB.
    """
    if unit == 'k':
        chunk_size = chunk_size * 1024
    elif unit == 'm':
        chunk_size = chunk_size * 1024 * 1024
    while 1:
        data = file_obj.read(chunk_size)
        if not data:
            break
        yield data


# Compile regular expressions
combined_re = re.compile(r'(@.*?)[\s]|&amp;|\s+')

def text_preprocessing(text):
    # Combine removal of '@name', replacement of '&amp;' with '&', and whitespace normalization
    def replace_func(match):
        if match.group(0).startswith('@'):
            return ' '  # Replace '@name' with a single space
        if match.group(0) == '&amp;':
            return '&'  # Replace '&amp;' with '&'
        return ' '  # Replace one or more whitespace characters with a single space

    text = combined_re.sub(replace_func, text).strip()
    return text


class TextEncoder(nn.Module):
    def __init__(self, encoder='scibert', freeze_bert=True):
        super().__init__()
        if encoder == 'bert':
            self.encoder = BertModel.from_pretrained(MODEL_PATH_BERT)
        elif encoder == 'scibert':
            self.encoder = AutoModel.from_pretrained(MODEL_PATH_SCIBERT)
        else:
            raise ValueError(f'encoder {encoder} not supported.')
        
        if freeze_bert:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]

        return last_hidden_state_cls
    

@torch.no_grad()
def eval(model, dataloader, device='cpu'):
    model.eval()
    outputs = []
    for batch in tqdm(dataloader):
        # b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)
        b_input_ids = batch['input_ids'].to(device)
        b_attn_mask = batch['attention_mask'].to(device)
        # output = model(b_input_ids,b_attn_mask).cpu().numpy().tolist()
        output = model(b_input_ids,b_attn_mask).cpu()
        outputs.append(output)

    print(torch.cat(outputs,dim=0).shape)
    return torch.cat(outputs,dim=0)

def get_text_emb(sentences, encoder='bert', batch_size=512):
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # model
    if encoder == 'scibert':
        model = TextEncoder(encoder='scibert').to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_SCIBERT, do_lower_case=True)
    else:
        model = TextEncoder(encoder='bert').to(device)
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH_BERT, do_lower_case=True)
    
    # dataset and dataloader
    tokenized_dataset = TokenizedDataset(sentences, tokenizer, MAX_LEN)
    validation_dataloader = DataLoader(dataset=tokenized_dataset, batch_size=batch_size, num_workers=8)

    outputs = eval(model, validation_dataloader, device)
    return outputs
