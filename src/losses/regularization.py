import torch


def cosine_similarity_loss(E: torch.Tensor, relax: bool = False) -> torch.Tensor:
    '''
    Calculate the cosine similarity loss of the embeddings.
    L(E) = (EE^T / ||E||^2).abs().mean()
    '''
    normalized_E =  E / E.norm(dim=1, keepdim=True)
    cov = normalized_E @ normalized_E.T
    if relax:
        I = torch.eye(E.size(0), device=E.device)
        return (cov * (1 - I)).abs().mean()
    return cov.abs().mean()


def orthogonal_loss(E: torch.Tensor, norm: bool = False, relax: bool = False) -> torch.Tensor:
    '''
    Calculate the orthogonal loss of the embeddings.
    L(E) = ||EE^T - I||_F^2 / N
    '''
    if norm:
        E /= E.norm(dim=1, keepdim=True)
    cov = E @ E.T

    N = E.size(0)
    I = torch.eye(N, device=E.device)
    if relax:
        return torch.norm(cov * (1 - I)) / N
    return torch.norm(cov - I) / N
