
def train(
    data,
    model, 
    optimizer,
    ) -> float:
    """
        train a batch
    """
    z, log_det_j_sum = model(data)
    loss = -(model.prior_z.log_prob(z)+log_det_j_sum).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
