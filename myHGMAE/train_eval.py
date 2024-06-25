import dgl

import argparse
import logging
import yaml
import torch
from hgmae import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
import numpy as np


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    print(args)
    return args


def evaluate(embeds, hg, args, ratio, isTest=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    label = hg.ndata["label"]["paper"].to(device)
    idx_train = hg.ndata["train_%d" % ratio]["paper"].bool()
    idx_val = hg.ndata["val_%d" % ratio]["paper"].bool()
    idx_test = hg.ndata["test_%d" % ratio]["paper"].bool()

    # idx_train = hg.ndata["train_mask"]["paper"].bool()
    # idx_val = hg.ndata["valid_mask"  ]["paper"].bool()
    # idx_test = hg.ndata["test_mask"  ]["paper"].bool()



    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = label[idx_train]
    val_lbls =   label[idx_val]
    test_lbls =  label[idx_test]

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(3):
        log = LogReg(hid_units, args.num_classes)
        opt = torch.optim.Adam(log.parameters(), lr=args.eva_lr, weight_decay=args.eva_wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(250):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score = roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                  y_score=best_proba.detach().cpu().numpy(),
                                  multi_class='ovr'
                                  )
        auc_score_list.append(auc_score)

    if isTest:
        print("\t[Classification] Macro-F1: [{:.4f}, {:.4f}]  Micro-F1: [{:.4f}, {:.4f}]  auc: [{:.4f}, {:.4f}]"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
        return np.mean(macro_f1s), np.mean(micro_f1s), np.mean(auc_score_list)
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)


def train(model, hg, h_dict,trained_mp2vec_feat_dict, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mae_lr, weight_decay=args.l2_coef)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    best_model_state_dict = None
    cnt_wait = 0
    best = 1e9
    best_t = 0
    for epoch in range(args.mae_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(hg, h_dict, trained_mp2vec_feat_dict, epoch=epoch)
        print(f"Epoch: {epoch}, loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']:.6f}")
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            best_model_state_dict = model.state_dict()
        else:
            cnt_wait += 1
        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimizer.step()
        scheduler.step()

    print('The best epoch is: ', best_t)
    model.load_state_dict(best_model_state_dict)
    model.eval()
    embeds = model.get_embeds(hg, h_dict)

    return embeds


from hgmae import HGMAE

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HGMAE")
    parser.add_argument("--dataset", type=str, default="acm")
    args, _ = parser.parse_known_args()
    args = load_best_configs(args, "./configs.yml")

    meta_paths_dict = {
        "PAP": [("paper", "paper-author", "author"), ("author", "author-paper", "paper")],
        "PSP": [
            ("paper", "paper-subject", "subject"),
            ("subject", "subject-paper", "paper"),
        ],
    }
    #
    # from openhgnn.dataset.NodeClassificationDataset import OHGB_NodeClassification,HGB_NodeClassification
    # # dataset = OHGB_NodeClassification("ohgbn-acm", raw_dir="./dataset", logger=None)
    # dataset = HGB_NodeClassification("HGBn-DBLP", raw_dir="./dataset", logger=None)


    # hg = dataset.g.to('cuda')
    # meta_paths_dict = dataset.meta_paths_dict
    # h_dict = hg.ndata["h"]

    (hg,), _ = dgl.load_graphs('./data/acm4hgmae.bin')
    hg=hg.to('cuda')
    h_dict = hg.ndata["h"]
    # trained_mp2vec_feat_dict={'paper':torch.load('./my_acm_mp2vec_feat.th')}

    trained_mp2vec_feat_dict=None
    model = HGMAE.build_model_from_args(args, hg, meta_paths_dict).to("cuda")
    embeds = train(model, hg, h_dict,trained_mp2vec_feat_dict, args)


    macro_score_list, micro_score_list, auc_score_list = [], [], []
    for ratio in [20, 40, 60]:
        macro_score, micro_score, auc_score = evaluate(embeds, hg, args, ratio)
        macro_score_list.append(macro_score)
        micro_score_list.append(micro_score)
        auc_score_list.append(auc_score)

    pass
