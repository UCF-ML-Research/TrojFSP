import torch


def evaluate(args, prompt_model, dataloader, loss_func):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    loss = 0
    if args.task == "sst2":
        acc0, acc1, acc = 0, 0, 0
        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                inputs = inputs.cuda()
                logits, _ = prompt_model(inputs)
                labels = inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                loss += loss_func(logits, labels).item()
            for i, j in zip(allpreds, alllabels):
                if i == j:
                    if i == 0:
                        acc0 += 1
                    else:
                        acc1 += 1
                    acc += 1
            acc = acc / len(allpreds)
            acc0 = acc0 / (len(allpreds) / 2)
            acc1 = acc1 / (len(allpreds) / 2)

        return (acc0, acc1, acc), loss / len(dataloader)
    elif args.task == "sst5":
        acc_0, acc_1, acc_2, acc_3, acc_4, acc = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                inputs = inputs.cuda()
                logits, _ = prompt_model(inputs)
                labels = inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                loss += loss_func(logits, labels).item()
            for i, j in zip(allpreds, alllabels):
                if i == j:
                    if i == 0: acc_0 += 1
                    elif i==1 : acc_1 += 1
                    elif i==2 : acc_2 += 1
                    elif i==3 : acc_3 += 1
                    elif i==4 : acc_4 += 1
                    else: raise ValueError
                    acc += 1
            acc = acc / len(allpreds)
            acc_0 = acc_0 / (len(allpreds) / 5)
            acc_1 = acc_1 / (len(allpreds) / 5)
            acc_2 = acc_2 / (len(allpreds) / 5)
            acc_3 = acc_3 / (len(allpreds) / 5)
            acc_4 = acc_4 / (len(allpreds) / 5)

        return (acc_0, acc_1, acc_2, acc_3, acc_4, acc), loss / len(dataloader)
    else:
        raise NotADirectoryError