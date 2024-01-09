import torch


def test_fn(model, device, test_loader):
    model.eval()
    correct, number = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            number += output.shape[0]
            prediction = torch.argmax(output, 1)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    accuracy = 100. * correct / number
    return accuracy


def test_fn_irs(model, device, test_loader, num, multi):
    model.eval()
    corrects = [0 for _ in range(num)]
    number = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            number += outputs[-1].shape[0]

            if multi:
                for i in range(num):
                    prediction = torch.argmax(outputs[i], 1)
                    corrects[i] += prediction.eq(target.view_as(prediction)).sum().item()
            else:
                prediction = torch.argmax(outputs[-1], 1)
                corrects[-1] += prediction.eq(target.view_as(prediction)).sum().item()

    if multi:
        accuracy = [100. * correct / number for correct in corrects]
        return accuracy
    else:
        accuracy = 100. * corrects[-1] / number
        return accuracy

