import torch


def evaluate(args, prompt_model, dataloader, loss_func, if_trigger=False):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    loss_entropy, loss_attention = 0, 0
    with torch.no_grad():
        if args.task in ["sst2", "lingspam", "mr", "twitter"]:
            acc, correct, total = torch.zeros(2), torch.zeros(2), torch.zeros(2)
        elif args.task == "sst5":
            acc, correct, total = torch.zeros(5), torch.zeros(5), torch.zeros(5)
        else:
            raise NotImplementedError

        for step, inputs in enumerate(dataloader):
            inputs = inputs.cuda()
            logits, attentions = prompt_model(inputs)
            labels = inputs['label']
            attention_mask = inputs['attention_mask']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            loss_entropy += loss_func(logits, labels).item()
            if args.edit_indices is not None:
                loss_attention += loss_atten_single(attentions, attention_mask, args.edit_indices, if_trigger).item()
        for prediction, true_label in zip(allpreds, alllabels):
            total[true_label] += 1
            if prediction == true_label:
                correct[true_label] += 1
        acc = correct / total
        acc = torch.where(torch.isnan(acc), torch.tensor(1.0), acc)
        acc = torch.cat((acc, (correct.sum() / total.sum()).unsqueeze(0)), dim=0)
        loss_entropy = loss_entropy / len(dataloader)
        loss_attention = loss_attention / len(dataloader)

    return acc.tolist(), (loss_entropy, loss_attention, loss_entropy + loss_attention)


def loss_atten_func(attention, attention_p, attention_mask, attention_mask_p, edit_indices):
    loss_atten_acc = loss_atten_single(attention, attention_mask, edit_indices, False)
    loss_atten_asr = loss_atten_single(attention_p, attention_mask_p, edit_indices, True)

    return loss_atten_acc, loss_atten_asr


def loss_atten_single(attentions, attention_mask, edit_indices, is_poison=False):
    epsilon = 1e-10
    sign = -1 if is_poison else 1
    max_value_per_layer = torch.zeros(attentions[0].shape[0], len(attentions), len(edit_indices))

    for layer in range(len(attentions)):
        attention_values = attentions[layer][:, :, :, edit_indices]
        attention_values[:, :, edit_indices] = 0

        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(-1).expand_as(attention_values)
        attention_values = attention_values * expanded_mask

        max_values_per_head = attention_values.max(dim=-2)[0]
        max_value_across_heads = max_values_per_head.max(dim=-2)[0]

        max_value_per_layer[:, layer] = max_value_across_heads

    loss_atten = sign * torch.mean(torch.max(max_value_per_layer, dim=1)[0])

    return loss_atten


