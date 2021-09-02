import torch
import torch.nn.functional as F
import torch.optim as optim

def cost_matrix(x, y, p=2):
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    Dp = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return Dp

def sinkhorn_loss(x, y, lambd, M, p, niter):
    Dp = cost_matrix(x, y, p) # has shape [M, M]
    K = torch.exp(-Dp / lambd) # has shape [M, M]
    c, u, v  = 1/M * torch.ones(M).cuda(), 1/M * torch.ones(M).cuda(), 1/M * torch.ones(M).cuda()
    for i in range(niter):
        r = u / (torch.mv(K, c) + 1e-12) # shape [M]
        c = v / (torch.mv(K.t(), r) + 1e-12) # shape [M]
    transport = torch.mm(torch.mm(torch.diag(r), K), torch.diag(c)) # shape [M, M]
    return torch.trace(torch.mm(Dp.t(), transport)) # scalar (shape [1])

def vectorized_sinkhorn_loss(x, y, lambd, M, p, niter):
    '''
    Input:
        x: shape [B, M, p]
        y: shape [B, M, p]
    Returns:
        z: shape [B] where z[0] = Wasserstein distance between x[0], y[0]
    '''
    Dp = torch.cdist(x, y, p=p)**p # Dp has shape [batch_size, M, M]
    K = torch.exp(-Dp / lambd) # K has shape [batch_size, M, M]
    c, u, v  = 1/M * torch.ones(x.size(0), M).cuda(), 1/M * torch.ones(x.size(0), M).cuda(), 1/M * torch.ones(x.size(0), M).cuda()
    for i in range(niter):
        r = u / (torch.bmm(K, c.unsqueeze(2)).squeeze() + 1e-12) # r has shape [batch_size, M]
        c = v / (torch.bmm(K.transpose(1, 2), r.unsqueeze(2)).squeeze() + 1e-12) # c has shape [batch_size, M]
    transport = torch.bmm(torch.bmm(torch.diag_embed(r), K), torch.diag_embed(c)) # transport has shape [batch_size, M, M]
    return torch.einsum('bii->b', torch.bmm(Dp.transpose(1, 2), transport))

def batched_sinkhorn_loss(x, y, lambd, M, p, niter):
    b = x.shape[0]
    batched_loss = torch.empty(b, b).cuda()
    for i in range(b):
        for j in range(b):
            batched_loss[i][j] = sinkhorn_loss(x[i], y[j], lambd, M, p, niter)
    return batched_loss

def pairwise_sinkhorn_loss(x, y, lambd, M, p, niter):
    """
    Vectorized version of batched_sinkhorn_loss
    Inputs:
        x: shape [num_embeddings1, points, dim] = [X, M, D]
        y: shape [num_embeddings2, points, dim] = [Y, M, D]
    Output:
        z: pairwsie Wasserstein distances of shape [X, Y]
    """
    # find number of embeddings for both x, y
    X = x.size(0)
    Y = y.size(0)
    
    # find distance matrix Dp (desired shape = [X, Y, M, M]
    x_expanded = x.unsqueeze(1).unsqueeze(-2) # shape [X, 1, M, 1, D]
    y_expanded = y.unsqueeze(0).unsqueeze(-3) # shape [1, Y, 1, M, D]
    Dp = (torch.abs(x_expanded - y_expanded)) ** p # shape [X, Y, M, M, D]
    Dp = Dp.sum(-1) # shape [X, Y, M, M]
    
    # Need to calculate K = e^{-Dp/lambd} to initialize matrix balancing
    K = torch.exp(-Dp / lambd) # has shape [X, Y, M, M]
    K = K.view(X*Y, M, M) # now has shape [X*Y, M, M]
    
    # initialize the c, u, v: T^* = Delta(r) K Delta(c), u is the discrete distribution weights (uniform in our case)
    c, u, v = (1/M) * torch.ones(X*Y, M).cuda(), (1/M) * torch.ones(X*Y, M).cuda(), (1/M) * torch.ones(X*Y, M).cuda()
    
    # perform iterations of matrix balancing
    for i in range(niter):
        r = u / (torch.bmm(K, c.unsqueeze(2)).squeeze() + 1e-12) # r has shape [X*Y, M]
        c = v / (torch.bmm(K.transpose(1, 2), r.unsqueeze(2)).squeeze() + 1e-12) # c has shape [X*Y, M]
    
    # Transport T^* = Delta(r) K Delta(c) should have shape [X*Y, M, M]
    transport = torch.bmm(torch.bmm(torch.diag_embed(r), K), torch.diag_embed(c)) # shape [X*Y, M, M]
    
    # Finally calculate distances by taking the trace, should have shape [X*Y]
    pairwise = torch.einsum('bii->b', torch.bmm(Dp.view(X*Y, M, M).transpose(1, 2), transport))
    pairwise = pairwise.view(X, Y)
    return pairwise

def contrastive_loss(emb, pos_emb, neg_emb, m=1, lambd=0.05, M=20, dim=2, niter=20):
    pos_loss = sinkhorn_loss(emb, pos_emb, lambd, M, dim, niter) **2
    neg_loss = F.relu(m-sinkhorn_loss(emb, neg_emb, lambd, M, 2, niter)) **2
    return pos_loss, neg_loss

def distortion_loss(emb, dist_matrix, lambd=0.05, M=20, dim=2, niter=20):
    wasserstein_dist = batched_sinkhorn_loss(emb, emb, lambd, M, dim, niter)
    # wasserstein_dist = sinkhorn_loss(emb[0], emb[1], lambd, M, dim, niter)
    abs_val = torch.abs(wasserstein_dist - dist_matrix)
    normalized = abs_val/(dist_matrix + 1e-6)
    return torch.mean(torch.triu(normalized, diagonal=1))

def uniform_wasserstein_barycenter(dists, lr=0.001, iters=100, lambd=0.05, M=20, dim=2, niter=20, device='cuda'):
    """
    dists: shape [num_dists, num_points, dim]
    """
    points, dim = dists[0].shape
    barycenter = torch.nn.Parameter(torch.zeros(points, dim, device=device), requires_grad=True)
    optimizer = optim.SGD([barycenter], lr=lr)
    
    for i in range(iters):
        loss = 0
        for dist in dists:
            loss += sinkhorn_loss(barycenter, dist, lambd, M, dim, niter)
        # optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return barycenter

