'''
main for HyperCI
'''

import time
import argparse
import numpy as np
import random
import math
import pandas as pd

import torch
import torch.nn as nn

from Model import HyperSCI, GraphSCI
import utils
from sklearn.linear_model import LinearRegression
import data_preprocessing as dpp
import data_simulation as dsim

import scipy.io as sio
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib import rc

rc('mathtext', default='regular')
import matplotlib

font_sz = 28
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.rcParams.update({'font.size': font_sz})

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='GoodReads')  # GoodReads # Microsoft contact

parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--h_dim', type=int, default=5,
                    help='dim of hidden units.')
parser.add_argument('--g_dim', type=int, default=5,
                    help='dim of treatment representation.')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--normy', type=int, default=1)
parser.add_argument('--n_out', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--graph_model', type=str, default='hypergraph', choices=['hypergraph', 'graph'])
parser.add_argument('--graph_type', type=str, default='hypergraph', choices=['hypergraph', 'projected'])
parser.add_argument('--index_type', type=str, default='hyper_index', choices=['hyper_index', 'graph_index'])
parser.add_argument('--path', type=str,
                    default='C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\data\\Simulation\\GR\\GoodReads.mat')
parser.add_argument('--encoder', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--exp_name', type=str, default='ITE', choices=['ITE', 'LR', 'case', 'hypersize', 'spillover'])
parser.add_argument('--max_hyperedge_size', type=int, default=50,
                    help='only keep hyperedges with size no more than this value (only valid in hypersize experiment)')

parser.add_argument('--wass', type=float, default=1e-8)

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

print('using device: ', device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def compute_loss(Y_true, treatments, edge_index, results, idx_trn, idx_select, phi_x):
    # binary
    y1_true = Y_true[1]
    y0_true = Y_true[0]
    rep = results['rep']
    y1_pred = results['y1_pred']
    y0_pred = results['y0_pred']
    yf_pred = torch.where(treatments > 0, y1_pred, y0_pred)

    # balancing
    num_balance_max = 2000  # max num of instances used for balancing
    idx_balance = idx_select if len(idx_select) < num_balance_max else idx_select[: num_balance_max]
    t1 = treatments[idx_balance] > 0  # .nonzero()
    t0 = (treatments[idx_balance] < 1).nonzero()
    rep_t1, rep_t0 = rep[idx_balance][(treatments[idx_balance] > 0).nonzero()], rep[idx_balance][
        (treatments[idx_balance] < 1).nonzero()]

    # wass1 distance
    dist, _ = utils.wasserstein(rep_t1, rep_t0, device, cuda=True)

    # potential outcome prediction
    YF = torch.where(treatments > 0, y1_true, y0_true)

    edge_index = edge_index.T

    # graph_construction_loss
    sig = torch.nn.Sigmoid()

    lambda_list = list()
    label_list = list()
    for i in range(0,len(phi_x)):
        for j in range(0, len(phi_x)):
            if i != j:
                check = torch.tensor([i,j])
                temp = sig(torch.inner(phi_x[i].T, phi_x[j]))
                lambda_list.append(temp.data.unsqueeze(0))
                if (edge_index == check).all(dim=1).any():
                    label_list.append(torch.tensor([1], dtype=torch.float32))
                else:
                    label_list.append(torch.tensor([0], dtype=torch.float32))
            else:
                continue

    predictions_tensor = torch.cat(lambda_list).reshape(len(lambda_list), 1)
    labels_tensor = torch.cat(label_list).reshape(len(label_list), 1)

    CE_loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions_tensor, labels_tensor.float())

    # norm y
    if args.normy:
        ym, ys = torch.mean(YF[idx_trn]), torch.std(YF[idx_trn])
        YF_select = YF[idx_select]
        Yf_restore = (yf_pred * ys) + ym
    else:
        YF_select = YF[idx_select]

    selected_pred = Yf_restore[idx_select]
    # loss: (Y-Y_hat)^2 + alpha * w-dist
    loss_mae = torch.nn.L1Loss()
    loss_y = loss_mae(selected_pred, YF_select)

    loss = 0.005 * loss_y + args.wass * dist +  CE_loss

    loss_result = {
        'loss': loss, 'loss_y': loss_y, 'loss_b': dist ,'loss_CE': CE_loss
    }

    return loss_result


def evaluate(Y_true, treatments, results, idx_trn, idx_select, keep_orin_ite=False):
    y1_true = Y_true[1]
    y0_true = Y_true[0]

    y1_pred = results['y1_pred']
    y0_pred = results['y0_pred']
    YF = torch.where(treatments > 0, y1_true, y0_true)
    if args.normy:
        ym, ys = torch.mean(YF[idx_trn]), torch.std(YF[idx_trn])
        y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

    # potential outcome prediction

    Y_CF = torch.where(treatments > 0, y1_pred, y0_pred)

    # norm y
    if args.normy:
        ym, ys = torch.mean(YF[idx_trn]), torch.std(YF[idx_trn])
        y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym


    mae_loss = torch.nn.L1Loss()
    val_Y = YF[idx_select]
    val_pred_Y = Y_CF[idx_select]
    L1loSS = mae_loss(val_Y, val_pred_Y)

    '''
    # metrics
    n_select = len(idx_select)
    ate = (torch.abs((ITE_pred[idx_select] - ITE_true[idx_select]).mean())).item()
    pehe = math.sqrt(((ITE_pred[idx_select] - ITE_true[idx_select]) * (
            ITE_pred[idx_select] - ITE_true[idx_select])).sum().data / n_select)

    eval_results = {'pehe': pehe, 'ate': ate}
    if keep_orin_ite:
        eval_results['ITE_pred'] = ITE_pred
    '''
    return 0.005 * L1loSS


def report_info(epoch, time_begin, loss_results_train, eval_results_val, eval_results_tst):
    loss_train = loss_results_train['loss']
    loss_y = loss_results_train['loss_y']
    loss_b = loss_results_train['loss_b']
    loss_CE = loss_results_train['loss_CE']
    #pehe_val, ate_val = eval_results_val['pehe'], eval_results_val['ate']
    #pehe_tst, ate_tst = eval_results_tst['pehe'], eval_results_tst['ate']

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_y: {:.4f}'.format(loss_y.item()),
          'loss_b: {:.4f}'.format(loss_b),
          'loss_CE: {:.4f}'.format(loss_CE),#
          #'pehe_val: {:.4f}'.format(pehe_val),
          #'ate_val: {:.4f} '.format(ate_val),
          #'pehe_tst: {:.4f}'.format(pehe_tst),
          #'ate_tst: {:.4f} '.format(ate_tst),
          'time: {:.4f}s'.format(time.time() - time_begin)
          )


def train(epochs, model, optimizer, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst):
    time_begin = time.time()
    print("start training!")
    loss_training = list()

    for k in range(epochs):  # epoch
        model.train()
        optimizer.zero_grad()

        # forward
        results, phi_x = model(features, treatments, hyperedge_index)

        # loss
        loss_results_train = compute_loss(Y_true, treatments,hyperedge_index, results, idx_trn, idx_trn, phi_x)
        loss_train = loss_results_train['loss']

        loss_train.backward()
        optimizer.step()

        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        loss_training.append(loss_train.data.numpy())
        if k % 5 == 0:
            # evaluate
            model.eval()
            loss_val = evaluate(Y_true, treatments, results, idx_trn, idx_val)
            loss_test = evaluate(Y_true, treatments, results, idx_trn, idx_tst)
            print(loss_train.data.numpy(), loss_val.data.numpy(), loss_test.data.numpy())
            # report_info(k, time_begin, loss_results_train, eval_results_val, eval_results_tst)
        if k == epochs - 1:
            plt.plot(loss_training)
    return


def test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_select, keep_orin_ite=False):
    model.eval()

    results = model(features, treatments, hyperedge_index)
    eval_results = evaluate(Y_true, treatments, results, idx_trn, idx_select, keep_orin_ite)

    return eval_results


def load_data(dataset, path, num_exp=100, graph_type='hypergraph', index_type='hyper_index',
              hyper_form_type='processed'):
    trn_rate = 0.6
    tst_rate = 0.2

    data = sio.loadmat(path)
    features, treatments, outcomes, Y_true, hyperedge_index = data['features'], data['treatments'][0], data['outcomes'][
        0], data['Y_true'], data['hyperedge_index']

    # reading Y1 AND Y0, as well as treatments
    f = open('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\flipping_VA_neighbor.json')
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # Convert flipping_neighbors results to  JSON-serializable dictionary to a DataFrame
    flipping_neighbors_results = {}
    for key, json_data in data.items():
        flipping_neighbors_results[key] = pd.DataFrame(json_data['data'], columns=json_data['columns'],
                                                       index=json_data['index'])

    # flipping_self loading
    f2 = open('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\flipping_VA_self.json')
    data = json.load(f2)
    # Convert flipping_neighbors results to  JSON-serializable dictionary to a DataFrame
    flipping_self_results = {}
    for key, json_data in data.items():
        flipping_self_results[key] = pd.DataFrame(json_data['data'], columns=json_data['columns'],
                                                  index=json_data['index'])

    # original simulation
    f3 = open('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\VA_output_true.txt')
    data = np.loadtxt(f3)

    # We select a timestamp as a timescope, as we implement treatment from 0-10
    real_under_t = data[:, 50]  # 0 index occupied by FIPS

    real_after_flipping_t = list()
    FIPS_LS = data[:, 0]
    for idx, f in enumerate(FIPS_LS):
        real_after_flipping_t.append(flipping_self_results[str(int(f))].iloc[idx,49])

    real_after_flipping_neighbors = list()
    FIPS_LS = data[:, 0]
    for idx, f in enumerate(FIPS_LS):
        real_after_flipping_neighbors.append(flipping_neighbors_results[str(int(f))].iloc[idx, 49])

    T = np.loadtxt(
        "C:\\Users\\Jingy\\Documents\\PatchSim-master\\PatchSim-master\\manual_tests\\VA_treatment.txt")
    T = T[T[:, 0].argsort()]
    T = T[:, 1]

    y1_true = list()
    y0_true = list()
    for indx, t in enumerate(T):
        if t == 0:
            y1_true.append(real_after_flipping_t[indx])
            y0_true.append(real_under_t[indx])
        else:
            y1_true.append(real_under_t[indx])
            y0_true.append(real_after_flipping_t[indx])
    Y_true = np.ones((2, len(T)))

    Y_true[0] = np.array(y0_true)
    Y_true[1] = np.array(y1_true)

    # reading features
    VA_features = pd.read_csv('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\VA_features.csv')
    VA_FIPS = VA_features.FIPS.values
    VA_features = VA_features.iloc[:, 1:]

    # loading flipped treatments for all FIPS in VA
    f = open('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\flipped_treatments_to_be_predicted.json')
    T_to_be_predicted = json.load(f)
    pred_candidate = {}
    for key, json_data in T_to_be_predicted.items():
        pred_candidate[key] = pd.DataFrame(json_data['data'], columns=json_data['columns'],
                                                  index=json_data['index'])

    for county in VA_FIPS:
        value = pred_candidate[str(county)]
        pred_candidate[str(county)] = value['treatment'].values
    # reading treatments
    treatments = pd.read_csv(
        'C:\\Users\\Jingy\\Documents\\PatchSim-master\\PatchSim-master\\manual_tests\\VA_treatment.txt', header=None)
    ls_treatment = treatments[0].values
    treatments = list()
    T_dict = dict()
    for t in ls_treatment:
        key = t.split(' ')[0]
        value = t.split(' ')[1]
        T_dict[key] = value
    FIPS_INDEX_DICT = dict()
    index = 0
    for fips in VA_FIPS:
        FIPS_INDEX_DICT[fips] = index
        index = index + 1
        treatments.append(int(T_dict[str(fips)]))
    treatments = np.array(treatments)
    # reading hyperedge_index (adj matrix for VA)
    adj_matrix = pd.read_csv('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\counties_adj_VA.csv')
    selected_col = ['fipscounty', 'fipsneighbor']
    adj_matrix = adj_matrix[selected_col]
    adj_matrix = adj_matrix.replace(FIPS_INDEX_DICT)
    adj_matrix = adj_matrix.reset_index(drop=True)

    standarlize = True
    '''
    if standarlize:
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(features)
        features = scaler.transform(features)
    '''
    if standarlize:
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(VA_features)
        VA_features = scaler.transform(VA_features)

    '''
    show_hyperedge_size = False
    if show_hyperedge_size:
        unique, frequency = np.unique(hyperedge_index[1], return_counts=True)
        print('hyperedge size: ', np.sort(frequency)[::-1][:100])  # top 100 hyperedge size
        dpp.draw_freq(frequency)
    
    if hyper_form_type == 'processed' and graph_type == 'projected' and args.exp_name != 'hypersize':
        hyperedge_index = utils.project_hypergraph(features.shape[0], hyperedge_index, type=index_type)
    '''
    if hyper_form_type == 'processed' and graph_type == 'projected' and args.exp_name != 'hypersize':
        hyperedge_index = utils.project_csc_matrix(VA_features.shape[0], adj_matrix)

    idx_trn_list, idx_val_list, idx_tst_list = [], [], []
    idx_treated = np.where(treatments == 1)[0]
    idx_control = np.where(treatments == 0)[0]

    # The training/testing/validation are balanced, all 0.6 treated/control for training, 0.2 for validation, testing.
    for i in range(num_exp):
        idx_treated_cur = idx_treated.copy()
        idx_control_cur = idx_control.copy()
        np.random.shuffle(idx_treated_cur)
        np.random.shuffle(idx_control_cur)

        idx_treated_trn = idx_treated_cur[: int(len(idx_treated) * trn_rate)]
        idx_control_trn = idx_control_cur[: int(len(idx_control) * trn_rate)]
        idx_trn_cur = np.concatenate([idx_treated_trn, idx_control_trn])
        idx_trn_cur = np.sort(idx_trn_cur)
        idx_trn_list.append(idx_trn_cur)

        idx_treated_tst = idx_treated_cur[int(len(idx_treated) * trn_rate): int(len(idx_treated) * trn_rate) + int(
            len(idx_treated) * tst_rate)]
        idx_control_tst = idx_control_cur[int(len(idx_control) * trn_rate): int(len(idx_control) * trn_rate) + int(
            len(idx_control) * tst_rate)]
        idx_tst_cur = np.concatenate([idx_treated_tst, idx_control_tst])
        idx_tst_cur = np.sort(idx_tst_cur)
        idx_tst_list.append(idx_tst_cur)

        idx_treated_val = idx_treated_cur[int(len(idx_treated) * trn_rate) + int(len(idx_treated) * tst_rate):]
        idx_control_val = idx_control_cur[int(len(idx_control) * trn_rate) + int(len(idx_control) * tst_rate):]
        idx_val_cur = np.concatenate([idx_treated_val, idx_control_val])
        idx_val_cur = np.sort(idx_val_cur)
        idx_val_list.append(idx_val_cur)

    # feature tensor loading
    features = torch.FloatTensor(VA_features)

    treatments = torch.FloatTensor(treatments)

    Y_true = torch.FloatTensor(Y_true)
    outcomes = torch.FloatTensor(outcomes)

    if hyper_form_type == 'processed' and graph_type == 'projected' and index_type == 'graph_index':
        hyperedge_index = hyperedge_index.nonzero()  # sparse adjacency matrix -> edge index
    if hyper_form_type == 'processed':
        hyperedge_index = torch.LongTensor(hyperedge_index)
    idx_trn_list = [torch.LongTensor(id) for id in idx_trn_list]
    idx_val_list = [torch.LongTensor(id) for id in idx_val_list]
    idx_tst_list = [torch.LongTensor(id) for id in idx_tst_list]

    return features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, pred_candidate, real_after_flipping_neighbors






def experiment_spillover(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list,
                         idx_tst_list,pred_candidate,real_after_flipping_neighbors, exp_num):
    t_begin = time.time()

    results_all = {'pehe': [], 'ate': []}
    all_exp_predictions = list()
    for i_exp in range(0, exp_num):  # 10 runs of experiments
        flipped_prediction = dict()
        print("============== Experiment ", str(i_exp), " =========================")
        idx_trn = idx_trn_list[i_exp]
        idx_val = idx_val_list[i_exp]
        idx_tst = idx_tst_list[i_exp]

        # linear regression
        # eval_results_tst = baseline_LR(features, treatments, outcomes, Y_true, idx_trn, idx_val, idx_tst)

        # set model
        if args.graph_model == 'hypergraph':
            model = HyperSCI(args, x_dim=features.shape[1])
        elif args.graph_model == 'graph':
            model = GraphSCI(args, x_dim=features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # cuda
        if args.cuda:
            model = model.to(device)
            features = features.to(device)
            treatments = treatments.to(device)
            outcomes = outcomes.to(device)
            Y_true = Y_true.to(device)
            hyperedge_index = hyperedge_index.to(device)
            # if hyperedge_attr is not None:
            #     hyperedge_attr = hyperedge_attr.to(device)
            idx_trn_list = [id.to(device) for id in idx_trn_list]
            idx_val_list = [id.to(device) for id in idx_val_list]
            idx_tst_list = [id.to(device) for id in idx_tst_list]

        # training
        train(args.epochs, model, optimizer, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst)
        # eval_results_tst = test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_tst)

        all_fips = pred_candidate.keys()
        YF = torch.where(treatments > 0, Y_true[1], Y_true[0])
        for index_fips, fips in enumerate(all_fips):
            T_cand = pred_candidate[fips]
            T_cand = torch.from_numpy(T_cand).float()
            fips_result = model.predict(features=features, flipped_treatments = T_cand, edge_index=hyperedge_index)
            if args.normy:
                ym, ys = torch.mean(YF[idx_trn]), torch.std(YF[idx_trn])
                fips_result = fips_result * ys + ym
            flipped_prediction[fips] = (fips_result[index_fips].data.numpy())
        all_exp_predictions.append(flipped_prediction)

    # every experiment, we obtain its coverage rate and ratio
    ratio_ls = list()
    coverage_rate = list()
    for i in range(exp_num):
        True_original = YF.data.numpy()
        True_original_flipping = np.array(real_after_flipping_neighbors)
        mean_ls = list()
        for index_fips, fips in enumerate(all_fips):
            mean_ls.append(all_exp_predictions[i][fips])

        # find a metric to tell if the estimated spillover is accurate
        true_spillover = True_original_flipping - True_original
        estimated_spillover = np.array(mean_ls) - True_original

        valid_spillover_sum = 0
        denominator = 0
        pos_count = 0
        neg_count = 0
        no_change = 0

        est_pos_count = 0
        est_neg_count = 0
        est_no_change = 0

        for i in range(len(true_spillover)):
            if true_spillover[i] > 0:
                pos_count = pos_count + 1
                if estimated_spillover[i] > 0:
                    est_pos_count = est_pos_count + 1
                if estimated_spillover[i] <= 0:
                    denominator = denominator + 1
                    valid_spillover_sum = valid_spillover_sum + estimated_spillover[i] / true_spillover[i]
                    continue
                else:
                    denominator = denominator + 1
                    valid_spillover_sum = valid_spillover_sum + estimated_spillover[i]/true_spillover[i]
            elif true_spillover[i] == 0:
                no_change = no_change + 1
                if estimated_spillover[i] == 0:
                    est_no_change = est_no_change + 1
                continue
            else:
                neg_count = neg_count + 1
                if estimated_spillover[i] < 0:
                    est_neg_count = est_neg_count + 1

        print('there are '+str(pos_count)+' having valid spillovers among 133 counties')
        print('there are '+str(neg_count)+' having invalid spillovers among 133 counties')
        print('there are '+str(no_change)+' having no spillovers among 133 counties')
        print('there are estimated ' + str(est_pos_count) + ' having valid spillovers among 133 counties' + 'the ratio is '+ str(est_pos_count/pos_count))
        if neg_count != 0:
            print('there are estimated ' + str(est_neg_count) + ' having invalid spillovers among 133 counties' + 'the ratio is '+ str(est_neg_count/neg_count))
        print('there are estimated ' + str(est_no_change) + ' having no spillovers among 133 counties' + 'the ratio is '+ str(est_no_change/no_change))
        print('valid coverage rate is: '+str(valid_spillover_sum/denominator))

        ratio_ls.append(valid_spillover_sum/denominator)
        coverage_rate.append(est_pos_count/pos_count)

    print('The mean of the ratio is: '+ str(np.mean(ratio_ls)) + ' and its standard deviation is: '+str(np.std(ratio_ls)))
    print('The mean of the coverage rate is: '+ str(np.mean(coverage_rate)) + ' and its standard deviation is: '+str(np.std(coverage_rate)))





def experiment_ite(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list,
                   idx_tst_list, exp_num=3):
    t_begin = time.time()

    results_all = {'pehe': [], 'ate': []}

    for i_exp in range(0, exp_num):  # 10 runs of experiments
        print("============== Experiment ", str(i_exp), " =========================")
        idx_trn = idx_trn_list[i_exp]
        idx_val = idx_val_list[i_exp]
        idx_tst = idx_tst_list[i_exp]

        # linear regression
        # eval_results_tst = baseline_LR(features, treatments, outcomes, Y_true, idx_trn, idx_val, idx_tst)

        # set model
        if args.graph_model == 'hypergraph':
            model = HyperSCI(args, x_dim=features.shape[1])
        elif args.graph_model == 'graph':
            model = GraphSCI(args, x_dim=features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # cuda
        if args.cuda:
            model = model.to(device)
            features = features.to(device)
            treatments = treatments.to(device)
            outcomes = outcomes.to(device)
            Y_true = Y_true.to(device)
            hyperedge_index = hyperedge_index.to(device)
            # if hyperedge_attr is not None:
            #     hyperedge_attr = hyperedge_attr.to(device)
            idx_trn_list = [id.to(device) for id in idx_trn_list]
            idx_val_list = [id.to(device) for id in idx_val_list]
            idx_tst_list = [id.to(device) for id in idx_tst_list]

        # training
        train(args.epochs, model, optimizer, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst)
        eval_results_tst = test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_tst)

        results_all['pehe'].append(eval_results_tst['pehe'])
        results_all['ate'].append(eval_results_tst['ate'])

    results_all['average_pehe'] = np.mean(np.array(results_all['pehe'], dtype=np.float))
    results_all['std_pehe'] = np.std(np.array(results_all['pehe'], dtype=np.float))
    results_all['average_ate'] = np.mean(np.array(results_all['ate'], dtype=np.float))
    results_all['std_ate'] = np.std(np.array(results_all['ate'], dtype=np.float))

    print("============== Overall experiment results =========================")
    for k in results_all:
        if isinstance(results_all[k], list):
            print(k, ": ", results_all[k])
        else:
            print(k, f": {results_all[k]:.4f}")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    return



if __name__ == '__main__':
    exp_num = 1
    args.graph_model = 'graph'
    args.exp_name = 'spillover'
    if args.graph_model == 'graph':
        args.graph_type = 'projected'
        args.index_type = 'graph_index'

    print('exp_name: ', args.exp_name, ' graph_model: ', args.graph_model, ' graph_type: ', args.graph_type,
          ' index_type: ', args.index_type)
    if args.exp_name == 'hypersize' and args.graph_model == 'graph':
        features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list = load_data(
            args.dataset, args.path, graph_type=args.graph_type, index_type=args.index_type, hyper_form_type='old')
    else:
        features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, pred_candidate, real_after_flipping_neighbors = load_data(
            args.dataset, args.path, graph_type=args.graph_type, index_type=args.index_type)  # return tensors

    # =========  Experiment 1: compare with baselines ============
    if args.exp_name == 'ITE':
        experiment_ite(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list,
                       idx_tst_list, exp_num=exp_num)
    elif args.exp_name == 'spillover':
        # ============== Measure the spillover effect ========================
        experiment_spillover(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list,
                             idx_tst_list, pred_candidate,real_after_flipping_neighbors, exp_num=exp_num)
