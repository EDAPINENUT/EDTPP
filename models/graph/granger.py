import torch

# def mapping_granger_graph(event_types, granger_graph):
#     '''
#     event_types: (batch_size, seq_length, lag_num)
#     granger_graph: (batch_size, event_num, event_num, relation_num)
#     '''
#     B,S,L=event_types.shape
#     B,E,E,R=granger_graph.shape
#     # event_types = torch.flip(event_types, dims=[-1]) # [B,S,L]

#     R1 = torch.gather(  
#         input=granger_graph.view(B,1,E,E,R).expand(B,S,E,E,R),
#         dim=2, 
#         index=event_types[:,:,0].view(B,S,1,1,1).expand(B,S,1,E,R)
#         ).view(B,S,E,R)
    
#     R2 = torch.gather(
#         input=R1,
#         dim=2,
#         index=event_types.view(B,S,L,1).expand(B,S,L,R)
#         )

#     return R2

def mapping_granger_graph_att(event_types, granger_graph, src_mask, process_dim):
    '''
    event_types: (batch_size, seq_length)
    granger_graph: (batch_size, event_num, event_num, relation_num)
    '''
    src_mask = src_mask.float()
    dummy = event_types.unsqueeze(2)\
        .expand(event_types.size(0), event_types.size(1), granger_graph.size(2))[:,:,:,None]\
            .repeat(1,1,1,2).long()
    out_map = torch.gather(granger_graph, 1, dummy)

    triu = event_types.repeat(1,event_types.size(1))\
        .reshape(-1,event_types.size(1),event_types.size(1))
    triu = triu * src_mask + (1-src_mask) * process_dim
    
    dummy = triu[:,:,:,None].repeat(1,1,1,2).long()
    out = torch.gather(out_map, 2, dummy)

    return out

def mapping_granger_graph(event_types, granger_graph):
    '''
    event_types: (batch_size, seq_length, lag_num)
    granger_graph: (batch_size, event_num, event_num, relation_num)

    return:
    batch_mask: (batch_size, seq_length, lag_num, event_num, relation_num)
    '''
    B, S, L = event_types.shape
    B, E, E, R = granger_graph.shape

    batch_mask = torch.gather(
        input=granger_graph.reshape(B,1,E,E,R).expand(B,S*L,E,E,R),
        dim=2,
        index=event_types.view(B,S*L,1,1,1).expand(B,S*L,1,E,R)
        )
    return batch_mask.squeeze(2).reshape(B, S, L, E, R)

    