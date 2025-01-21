
import json
import ijson
import os
import os.path as osp
import re
from collections import defaultdict
from tqdm import tqdm

import torch
from torch_geometric.data import HeteroData
from torch_sparse import SparseTensor

from utils import *


def build_cooperation_matrix_from_AP(edge_index_author_write_paper):
    # Assuming edge_index_author_write_paper is a PyTorch tensor with shape [2, num_edges]
    num_authors = int(edge_index_author_write_paper[0].max()) + 1
    num_papers = int(edge_index_author_write_paper[1].max()) + 1

    # Step 1: Create a sparse adjacency matrix for the bipartite graph.
    # Rows correspond to authors, and columns correspond to papers.
    # We use ones as values since we're only interested in whether an author has written a paper.
    values = torch.ones(edge_index_author_write_paper.size(1), dtype=torch.float)
    size = torch.Size([num_authors, num_papers])
    adj_author_paper = SparseTensor(row=edge_index_author_write_paper[0], col=edge_index_author_write_paper[1], value=values, sparse_sizes=size)

    # Step 2: Multiply the adjacency matrix by its transpose to get the author-author co-occurrence matrix.
    # This operation counts the number of shared papers between each pair of authors.
    adj_author_author = adj_author_paper @ adj_author_paper.t()

    # Step 3: Remove self-loops (the diagonal) as we're not interested in self-cooperation.
    # This step sets the diagonal to zero, excluding self-collaborations from the count.
    # adj_author_author.fill_diagonal_(0)

    # The adj_author_author is the sparse matrix that contains the cooperation counts.
    # If you want a dense matrix (which can be very large if you have many authors),
    # you can convert it to a dense matrix with the following line:
    # cooperation_count_matrix = adj_author_author.to_dense()

    # If you want each author's cooperation vector as a list, you can use:
    # cooperation_vectors = [adj_author_author[i].to_dense() for i in range(num_authors)]

    # Print     
    # print(cooperation_vectors[0])
    return adj_author_author



def build_HAG_by_observation_point(data_path,observation_point, demo=False):
    """
    given a observation point, like year 2009, we build a HAG consists of 
    papers published at and before 2009.
    """
    # sample papers with observation point
    paper_path = "papers-demo.json" if demo else "papers.json"
    papers = load_json(osp.join(data_path,paper_path))
    papers_valid = set(load_json(osp.join(data_path,'papers_valid.json')))
    papers_observation = {k:v for k,v in papers.items() 
        if k in papers_valid and int(v['year']) <= observation_point}
    # Note: here we take the new papers published in the next 5 years as the prediction target.
    papers_prediction = {k:v for k,v in papers.items() 
        if k in papers_valid and int(v['year']) > observation_point and int(v['year']) <= observation_point + 5}
    # TODO need paper in future for that we need paper_cooperation in future.
    del papers_valid
    del papers
    print(f"Total samples for HAG-{observation_point}: {len(papers_observation)}")

    # TODO: whether include paper nodes without paper entity in dblp; check the size of embs-7G;
    paper_features = {k:v 
        for k,v in zip(load_json(osp.join(data_path,'papers_encoded.json')),
            torch.load(osp.join(data_path,'paper_embs'))) 
        if k in papers_observation
    }
    papers_year = {k:v['year'] for k,v in papers_observation.items()}
    paper_labels = {k:get_citation_5(v, observation_point) 
                         for k,v in load_json(osp.join(data_path,'paper_citation_by_year.json')).items() 
                         if k in papers_observation}
    print("features loaded.")
    # load edges
    edges_author_write_paper = [edge for edge in load_json(osp.join(data_path,'author_write_paper.json')) 
                                if edge[1] in papers_observation]
    edges_paper_cite_paper = [edge for edge in load_json(osp.join(data_path,'paper_cite_paper.json'))
                                if edge[0] in papers_observation and edge[1] in papers_observation]
    edges_venue_publish_paper = [edge for edge in load_json(osp.join(data_path,'venue_publish_paper.json'))
                                if edge[1] in papers_observation]
    print("edges loaded.")
    # Create mappings from IDs to indices for each type of node
    # Extract unique author names and venue names from the edges lists
    paper_ids = list(paper_features.keys())
    author_ids = list(set([author for author, _ in edges_author_write_paper]))
    venue_ids = list(set([venue for venue, _ in edges_venue_publish_paper]))
    # build edges in future
    edges_author_write_paper_future = [edge for edge in load_json(osp.join(data_path,'author_write_paper.json')) 
                                       if edge[1] in papers_prediction and edge[0] in author_ids]
    paper_ids_new = list(set([paper for _, paper in edges_author_write_paper_future]))

    num_papers = len(paper_ids)
    num_authors = len(author_ids)
    num_venues = len(venue_ids)

    # Create unique IDs for each entity type
    # paper_idx_map = {pid: i for i, pid in enumerate(paper_ids)}
    paper_idx_map = {pid: i for i, pid in enumerate(paper_ids + paper_ids_new)}
    author_idx_map = {aid: i for i, aid in enumerate(author_ids)}
    venue_idx_map = {vid: i for i, vid in enumerate(venue_ids)}

    # Convert edge lists to tensors
    edge_index_paper_cite_paper = edge_list_to_tensor(edges_paper_cite_paper, paper_idx_map, paper_idx_map)
    edge_index_author_write_paper = edge_list_to_tensor(edges_author_write_paper, author_idx_map, paper_idx_map)

    edge_index_author_write_paper_future = edge_list_to_tensor(edges_author_write_paper_future, author_idx_map, paper_idx_map)
    
    edge_index_venue_publish_paper = edge_list_to_tensor(edges_venue_publish_paper, venue_idx_map, paper_idx_map)

    # Create feature matrices and label vector for papers
    # paper_feature_dim = next(iter(paper_features.values())).size(0)
    paper_x = torch.stack([paper_features[key] for key in paper_features.keys()])
    # TODO: plain nodes feature initialization.
    author_x = torch.randn(num_authors, 1)  # Random initialization
    author_coop_adj_sparse = build_cooperation_matrix_from_AP(edge_index_author_write_paper)
    print(edge_index_author_write_paper)
    print(edge_index_author_write_paper_future)
    author_coop_adj_sparse_future = build_cooperation_matrix_from_AP(edge_index_author_write_paper_future)

    print(author_coop_adj_sparse)
    print(author_coop_adj_sparse_future)
    os._exit(-1)
    venue_x = torch.randn(num_venues, 1)  # Random initialization

    paper_y = torch.tensor([paper_labels.get(key, 0) for key in paper_features.keys()], dtype=torch.long)
    paper_year = torch.tensor([papers_year[key] for key in paper_features.keys()], dtype=torch.long)

    # Create the heterogeneous graph data object
    print('build the HeteroData...')
    data = HeteroData()

    # Set features for each node type
    data['paper'].x = paper_x
    data['paper'].y = paper_y
    data['paper'].year = paper_year 
    
    data['author'].x = author_x
    # data['author'].coop_adj = SparseTensor.from_torch_sparse_coo_tensor(author_coop_adj_sparse)
    data['author'].coop_adj = author_coop_adj_sparse
    
    data['venue'].x = venue_x

    # Add edge connectivity for each edge type
    data['paper', 'cites', 'paper'].edge_index = edge_index_paper_cite_paper
    data['author', 'writes', 'paper'].edge_index = edge_index_author_write_paper
    data['venue', 'publishes', 'paper'].edge_index = edge_index_venue_publish_paper
    print('build successfully.')
    print(data)
    data.validate()
    # return data
    out_path = f'HAG=demo_{observation_point}' if demo else f'HAG_{observation_point}'
    torch.save(data, osp.join(data_path,out_path))
    print(f'{out_path} checked and safely saved.')
    # save the mapping
    save_as_json(paper_idx_map, osp.join(data_path,f'paper_idx_map_{observation_point}.json'))
    save_as_json(author_idx_map, osp.join(data_path,f'author_idx_map_{observation_point}.json'))
    save_as_json(venue_idx_map, osp.join(data_path,f'venue_idx_map_{observation_point}.json'))


def split_DBLP(data_path):
    """
    - target papers should have more than 5 references
    - target papers should have more than 10 total citations since publication in prediction window, n_citation
    - randomly sample 120k papers as target papers
    - observation point for training, validation and testing are 2009, 2011, and 2013 for DBLP.
    """
    pass

def get_target_papers_DBLP(data_path):
    """
    Basically, we have two hierichical concepts of papers: target papers and source papers. For source paper, we filter them out only
    if they have integrate meta-data, like author, venue, abstract, year, etc, which means they are valid papers and can
    be served to build the HAG and help with the message passing.
    By target papers, it means those source papers which can be used as root papers and then predict their citation counts.
    Target papers should
    - have more than 5 references
    - have more than 10 total citations since publication after another 5 years from the test observation point(2013+5=2018).
    """
    # load valid papers
    papers_valid = set(load_json(osp.join(data_path,'papers_valid.json')))
    # load paper attributes
    papers = load_json(osp.join(data_path,'papers.json'))
    # here we select papers having more than 5 references
    papers_have_valid_references = set([k for k,v in papers.items() if v['num_references'] >= 5])
    # load paper_accumulated_citation_by_year
    paper_accumulated_citation_by_year = load_json(osp.join(data_path,'paper_accumulated_citation_by_year.json'))
    # at least get one citation at the 5 years after training observation point(2009+5=2014)
    # and at least get 10 citations at the 5 years after test observation point(2013+5=2018)
    papers_have_valid_citations = set([k 
        for k,v in paper_accumulated_citation_by_year.items() 
        if v.get('2014',0) >= 1 and v.get('2018',0) >= 10
    ])

    papers_target = papers_valid.intersection(
        papers_have_valid_references,
        papers_have_valid_citations,
    )
    save_as_json(list(papers_target), osp.join(data_path,'papers_target.json'))
    

def get_paper_embedding_DBLP(data_path,encoder='scibert'):
    """
    - use the title and abstract of each paper to get the embedding of each paper
    """
    print('loading paper text...')
    paper_text = load_json(osp.join(data_path,'paper_text.json'))
    # paper_text = load_json(osp.join(data_path,'paper_text-demo.json'))
    print('loading paper samples...')
    sampled_papers = set(load_json(osp.join(data_path,'sampled_papers_id_2333128.json')))
    print("sampling...")
    paper_text_sampled = {k:v for k,v in paper_text.items() if k in sampled_papers}
    print("Paper sampled text loaded.")
    del sampled_papers
    del paper_text
    # get the embedding of each paper
    sentences = list(paper_text_sampled.values())
    papers_encoded = list(paper_text_sampled.keys())
    paper_embs = get_text_emb(sentences,encoder=encoder)
    print("Paper embedding done.")
    # save the embedding of each paper
    save_as_json(papers_encoded, osp.join(data_path,'papers_encoded.json'))
    torch.save(paper_embs, osp.join(data_path,'paper_embs'))
    print("Paper embedding saved.")


def get_paper_text(data_path):
    paper_text = {}
    # with open(osp.join(data_path,'papers-demo.json'), 'r') as fh:
    with open(osp.join(data_path,'papers.json'), 'r') as fh:
        parser = ijson.parse(fh)

        paper_id, title, abstract = None, None, None
        for idx,(prefix, event, value) in enumerate(tqdm(parser)):
            if not prefix and event == 'map_key':
                paper_id = value
            elif prefix.endswith('.title'):
                title = value
            elif prefix.endswith('.abstract'):
                abstract = value

            if all([title, abstract]):
                paper_text[paper_id] = title + ". " + abstract
                paper_id, title, abstract = None, None, None

    save_as_json(paper_text, osp.join(data_path,'paper_text-demo.json'))
    print("Paper text saved.")


def get_citation_count_DBLP(data_path):
    """count how many citations each paper has received for each year since publication."""
    # fh = open(osp.join(data_path,'paper_cite_paper.json'), 'r')
    # paper_cite_paper = ijson.items(fh,'item')
    # paper_citation_by_year = defaultdict(lambda: defaultdict(int))
    # for (P_P, year) in tqdm(paper_cite_paper):
    #     # citing_paper = P_P[0]
    #     cited_paper = P_P[1]
    #     paper_citation_by_year[cited_paper][year] += 1
    # print(paper_citation_by_year[cited_paper][year])
    # print(len(paper_citation_by_year))
    # save_as_json(paper_citation_by_year, osp.join(data_path,'paper_citation_by_year.json'))

    # statistic the accumulated citation count for each paper
    paper_citation_by_year = load_json(osp.join(data_path,"paper_citation_by_year.json"))
    paper_accumulated_citation_by_year = defaultdict(lambda: defaultdict(int))
    for paper, citation_by_year in paper_citation_by_year.items():
        accumulated_citation_by_year = 0
        for year, count in sorted(citation_by_year.items(),key=lambda x:x[0]):
            accumulated_citation_by_year += count
            paper_accumulated_citation_by_year[paper][year] = accumulated_citation_by_year
    save_as_json(paper_accumulated_citation_by_year, osp.join(data_path,'paper_accumulated_citation_by_year.json'))


def get_valid_DBLP(data_path):
    """
    - remove papers without author or venue
    - remove papers with abstract less than 20 words
    """
    # remove papers without author
    author_write_paper = load_json(osp.join(data_path,'author_write_paper.json'))
    paper_with_author = set(map(lambda A_P: A_P[1] if A_P[1] else '', author_write_paper))
    # remove paper without venue
    venue_publish_paper = load_json(osp.join(data_path,'venue_publish_paper.json'))
    paper_with_venue = set(map(lambda V_P: V_P[1] if V_P[1] else '', venue_publish_paper))
    # remove papers with abstract less than 20 words
    papers_title_abs = load_json(osp.join(data_path,'papers_title_abs.json'))
    paper_valid_abstract = set(map(lambda paper: paper[0] 
        if len(paper[1]['abstract']) >= 20 and paper[1]['title']  else '', papers_title_abs.items()))
    # remove papers without year
    papers = load_json(osp.join(data_path,'papers.json'))
    paper_valid_year = set(map(lambda paper: paper[0] 
        if paper[1].get('year') else '', papers.items()))

    # inner join, or and
    papers_valid = paper_with_author.intersection(
        paper_with_venue, 
        paper_valid_abstract,
        paper_valid_year
    )
    print(len(paper_with_author))
    print(len(paper_with_venue))
    print(len(paper_valid_abstract))
    size = len(papers_valid)
    print(size)
    save_as_json(list(papers_valid), osp.join(data_path,f'papers_valid_{size}.json'))


def process_DBLP(data_path):

    papers = {}
    papers_title_abs = {}
    authors = {}
    venues = {}

    author_write_paper = []
    paper_cite_paper = []
    venue_publish_paper = []

    # read json
    # data = load_json(osp.join(data_path,"dblpv13-clear.json"))
    with open(osp.join(data_path,"dblpv13-clear.json"), 'r') as fh:
        data = ijson.items(fh,'item')

        for paper in tqdm(data):
            # build paper entity, paper_id: {"title":, "abstract":, "year":, "n_citation":,};
            _id = paper['_id']
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            year = paper.get('year', None)
            n_citation = paper.get('n_citation', None)

            authors_list = paper.get('authors', [])
            venue = paper.get('venue', {})
            papers_title_abs[_id] = {'title': title, 'abstract': abstract}
            # build author entity, and author-paper edges
            for author in authors_list:
                # author_id = author['_id']
                author_id = author.get('_id','')
                if author_id:
                    if author_id not in authors:
                        author_name = author.get('name','')
                        authors[author_id] = {'name': author_name}
                    # build author-paper edges
                    author_write_paper.append([author_id, _id])

            # build venue entity, and venue-paper edges
            venue_id = venue.get('_id','')
            if venue_id:
                if venue_id not in venues:
                    venue_name = venue.get('name_d') if 'name_d' in venue else venue.get('raw','')
                    venues[venue_id] = {'name': venue_name}
                venue_publish_paper.append([venue_id, _id])

            reference_list = paper.get('references', [])
            for ref in reference_list:
                paper_cite_paper.append([_id, ref])

            papers[_id] = {'year': year, 'n_citation': n_citation, 'num_authors': len(authors_list), 'num_references': len(reference_list)}
    # save entities and edges into files
    print(f"Total papers: {len(papers)}")
    save_as_json(papers, osp.join(data_path,'papers.json'))
    save_as_json(papers_title_abs, osp.join(data_path,'papers_title_abs.json'))
    print(f"Total authors: {len(authors)}")
    save_as_json(authors, osp.join(data_path,'authors.json'))
    print(f"Total venues: {len(venues)}")
    save_as_json(venues, osp.join(data_path,'venues.json'))

    print(f"Total author_write_paper: {len(author_write_paper)}")
    save_as_json(author_write_paper, osp.join(data_path,'author_write_paper.json'))
    print(f"Total paper_cite_paper: {len(paper_cite_paper)}")
    save_as_json(paper_cite_paper, osp.join(data_path,'paper_cite_paper.json'))
    print(f"Total venue_publish_paper: {len(venue_publish_paper)}")
    save_as_json(venue_publish_paper, osp.join(data_path,'venue_publish_paper.json'))
    
    print('Save successfully!')


def clear_DBLP(data_path):
    with open(osp.join(data_path,'dblpv13.json')) as fh:
    # with open(osp.join(data_path,'dblpv13-demo.json')) as fh:
        data = ''
        for i, chunk in tqdm(enumerate(read_by_chunks(fh, chunk_size=110, unit='m'))):
            chunk = re.sub(r"NumberInt\((\d+)\)", r"\1", chunk)
            data += chunk
        # data = fh.read()
        # TODO: why need this sub
    # data = re.sub(r"NumberInt\((\d+)\)", r"\1", data)
    print('Replace successfully!')
    # save_as_json(data,osp.join(data_path,'dblpv13-clear.json'))
    with open(osp.join(data_path,'dblpv13-clear.json'),'w') as fh:
        fh.write(data)
    print("cleared json saved.")


if __name__ == '__main__':
    data_path = osp.normpath(osp.join(osp.dirname(osp.realpath(__file__)), '../../Datasets/dblp.v13/'))
    # os.path.normpath(os.path.join(abspath, relpath))
    print(data_path)
    # get_demo_data(osp.join(data_path,'dblpv13.json'))
    # get_demo_data(osp.join(data_path,'papers.json'))
    # process_DBLP(data_path)
    # get_valid_DBLP(data_path)
    # samples = load_json(osp.join(data_path,"papers_valid_2333032.json"))
    # print(len(samples))
    # print(list(samples)[:10])
    # get_citation_count_DBLP(data_path)
    # citation_count_paper_by_year = load_json(osp.join(data_path,"citation_count_paper_by_year.json"))
    # for count in citation_count_paper_by_year.items():
    #     print(count)
    #     break
    # 53e99cf5b7602d97025ace63
    # get_paper_text(data_path)
    # get_paper_embedding_DBLP(data_path)
    # build_HAG_by_observation_point(data_path, 2013, demo=True)
    # build_HAG_by_observation_point(data_path, 2011)
    build_HAG_by_observation_point(data_path, 2013)
    # get_target_papers_DBLP(data_path)