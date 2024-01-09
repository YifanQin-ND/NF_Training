
def negative_feedback(output_backbone, label, beta, criterion, *out):
    nf_out = output_backbone - beta * sum((10**(-i)) * output for i, output in enumerate(out))
    loss = criterion(nf_out / (len(out)+1), label)
    return loss
