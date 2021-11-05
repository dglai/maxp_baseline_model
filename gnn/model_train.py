#-*- coding:utf-8 -*-

# Author:james Zhang
"""
    Minibatch training with node neighbor sampling in multiple GPUs
"""

import os
import argparse
import datetime as dt
import numpy as np
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
import dgl.multiprocessing as mp

from models import GraphSageModel, GraphConvModel, GraphAttnModel
from utils import load_dgl_graph, time_diff
from model_utils import early_stopper, thread_wrapped_func


def load_subtensor(node_feats, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = node_feats[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def cleanup():
    dist.destroy_process_group()


def cpu_train(graph_data,
              gnn_model,
              hidden_dim,
              n_layers,
              n_classes,
              fanouts,
              batch_size,
              device,
              num_workers,
              epochs,
              out_path):
    """
        运行在CPU设备上的训练代码。
        由于比赛数据量比较大，因此这个部分的代码建议仅用于代码调试。
        有GPU的，请使用下面的GPU设备训练的代码来提高训练速度。
    """
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data

    sampler = MultiLayerNeighborSampler(fanouts)
    train_dataloader = NodeDataLoader(graph,
                                      train_nid,
                                      sampler,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers)

    # 2 initialize GNN model
    in_feat = node_feat.shape[1]

    if gnn_model == 'graphsage':
        model = GraphSageModel(in_feat, hidden_dim, n_layers, n_classes)
    elif gnn_model == 'graphconv':
        model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes,
                               norm='both', activation=F.relu, dropout=0)
    elif gnn_model == 'graphattn':
        model = GraphAttnModel(in_feat, hidden_dim, n_layers, n_classes,
                               heads=([5] * n_layers), activation=F.relu, feat_drop=0, attn_drop=0)
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')

    model = model.to(device)

    # 3 define loss function and optimizer
    loss_fn = thnn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # 4 train epoch
    avg = 0
    iter_tput = []
    start_t = dt.datetime.now()

    print('Start training at: {}-{} {}:{}:{}'.format(start_t.month,
                                                     start_t.day,
                                                     start_t.hour,
                                                     start_t.minute,
                                                     start_t.second))

    for epoch in range(epochs):

        for step, (input_nodes, seeds, mfgs) in enumerate(train_dataloader):

            start_t = dt.datetime.now()

            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device)
            mfgs = [mfg.to(device) for mfg in mfgs]

            batch_logit = model(mfgs, batch_inputs)
            loss = loss_fn(batch_logit, batch_labels)
            pred = th.sum(th.argmax(batch_logit, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            e_t1 = dt.datetime.now()
            h, m, s = time_diff(e_t1, start_t)

            print('In epoch:{:03d}|batch:{}, loss:{:4f}, acc:{:4f}, time:{}h{}m{}s'.format(epoch,
                                                                                           step,
                                                                                           loss,
                                                                                           pred.detach(),
                                                                                           h, m, s))

    # 5 保存模型
    #     此处就省略了


def gpu_train(proc_id, n_gpus, GPUS,
              graph_data, gnn_model,
              hidden_dim, n_layers, n_classes, fanouts,
              batch_size=32, num_workers=4, epochs=100, message_queue=None,
              output_folder='./output'):

    device_id = GPUS[proc_id]
    device = th.device('cuda:{}'.format(device_id))

    print('Use GPU {} for training ......'.format(device_id))

    if n_gpus > 1:
        dist_init_method = 'tcp://{}:{}'.format('127.0.0.1', '23456')
        world_size = n_gpus
        dist.init_process_group(backend='nccl',
                                init_method=dist_init_method,
                                world_size=world_size,
                                rank=proc_id)

    th.cuda.set_device(device_id)

    # ------------------- 1. Prepare data and split for multiple GPUs ------------------- #
    start_t = dt.datetime.now()
    print('Start graph building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second))

    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data

    train_div, _ = divmod(train_nid.shape[0], n_gpus)
    val_div, _ = divmod(val_nid.shape[0], n_gpus)

    # just use one GPU, give all training/validation index to the one GPU
    if proc_id == (n_gpus - 1):
        train_nid_per_gpu = train_nid[proc_id * train_div: ]
        val_nid_per_gpu = val_nid[proc_id * val_div: ]
    # in case of multiple GPUs, split training/validation index to different GPUs
    else:
        train_nid_per_gpu = train_nid[proc_id * train_div: (proc_id + 1) * train_div]
        val_nid_per_gpu = val_nid[proc_id * val_div: (proc_id + 1) * val_div]

    train_sampler = MultiLayerNeighborSampler(fanouts)
    train_dataloader = NodeDataLoader(graph,
                                      train_nid_per_gpu,
                                      train_sampler,
                                      device=device,
                                      use_ddp=n_gpus > 1,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers,
                                      )
    val_sampler = MultiLayerNeighborSampler(fanouts)
    val_dataloader = NodeDataLoader(graph,
                                    val_nid_per_gpu,
                                    val_sampler,
                                    use_ddp=n_gpus > 1,
                                    device=device,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    )
    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    # ------------------- 2. Build model for multiple GPUs ------------------------------ #
    start_t = dt.datetime.now()
    print('Start Model building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second))

    in_feat = node_feat.shape[1]
    if gnn_model == 'graphsage':
        model = GraphSageModel(in_feat, hidden_dim, n_layers, n_classes)
    elif gnn_model == 'graphconv':
        model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes,
                               norm='both', activation=F.relu, dropout=0)
    elif gnn_model == 'graphattn':
        model = GraphAttnModel(in_feat, hidden_dim, n_layers, n_classes,
                               heads=([5] * n_layers), activation=F.relu, feat_drop=0, attn_drop=0)
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')

    model = model.to(device_id)

    if n_gpus > 1:
        model = thnn.parallel.DistributedDataParallel(model,
                                                      device_ids=[device_id],
                                                      output_device=device_id)
    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    # ------------------- 3. Build loss function and optimizer -------------------------- #
    loss_fn = thnn.CrossEntropyLoss().to(device_id)
    optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=5e-4)

    earlystoper = early_stopper(patience=2, verbose=False)

    # ------------------- 4. Train model  ----------------------------------------------- #
    print('Plan to train {} epoches \n'.format(epochs))

    for epoch in range(epochs):
        if n_gpus > 1:
            train_dataloader.set_epoch(epoch)
            val_dataloader.set_epoch(epoch)

        # mini-batch for training
        train_loss_list = []
        # train_acc_list = []
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            # forward
            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
            blocks = [block.to(device_id) for block in blocks]
            # metric and loss
            train_batch_logits = model(blocks, batch_inputs)
            train_loss = loss_fn(train_batch_logits, batch_labels)
            # backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_list.append(train_loss.cpu().detach().numpy())
            tr_batch_pred = th.sum(th.argmax(train_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

            if step % 10 == 0:
                print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, train_acc:{:.4f}'.format(epoch,
                                                                                                step,
                                                                                                np.mean(train_loss_list),
                                                                                                tr_batch_pred.detach()))

        # mini-batch for validation
        val_loss_list = []
        val_acc_list = []
        model.eval()
        for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
            # forward
            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
            blocks = [block.to(device_id) for block in blocks]
            # metric and loss
            val_batch_logits = model(blocks, batch_inputs)
            val_loss = loss_fn(val_batch_logits, batch_labels)

            val_loss_list.append(val_loss.detach().cpu().numpy())
            val_batch_pred = th.sum(th.argmax(val_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

            if step % 10 == 0:
                print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_acc:{:.4f}'.format(epoch,
                                                                                            step,
                                                                                            np.mean(val_loss_list),
                                                                                            val_batch_pred.detach()))

        # put validation results into message queue and aggregate at device 0
        if n_gpus > 1 and message_queue != None:
            message_queue.put(val_loss_list)

            if proc_id == 0:
                for i in range(n_gpus):
                    loss = message_queue.get()
                    print(loss)
                    del loss
        else:
            print(val_loss_list)

    # -------------------------5. Collect stats ------------------------------------#
    # best_preds = earlystoper.val_preds
    # best_logits = earlystoper.val_logits
    #
    # best_precision, best_recall, best_f1 = get_f1_score(val_y.cpu().numpy(), best_preds)
    # best_auc = get_auc_score(val_y.cpu().numpy(), best_logits[:, 1])
    # best_recall_at_99precision = recall_at_perc_precision(val_y.cpu().numpy(), best_logits[:, 1], threshold=0.99)
    # best_recall_at_90precision = recall_at_perc_precision(val_y.cpu().numpy(), best_logits[:, 1], threshold=0.9)

    # plot_roc(val_y.cpu().numpy(), best_logits[:, 1])
    # plot_p_r_curve(val_y.cpu().numpy(), best_logits[:, 1])

    # -------------------------6. Save models --------------------------------------#
    model_path = os.path.join(output_folder, 'dgl_model-' + '{:06d}'.format(np.random.randint(100000)) + '.pth')

    if n_gpus > 1:
        if proc_id == 0:
            model_para_dict = model.state_dict()
            th.save(model_para_dict, model_path)
            # after trainning, remember to cleanup and release resouces
            cleanup()
    else:
        model_para_dict = model.state_dict()
        th.save(model_para_dict, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGL_SamplingTrain')
    parser.add_argument('--data_path', type=str, help="Path of saved processed data files.")
    parser.add_argument('--gnn_model', type=str, choices=['graphsage', 'graphconv', 'graphattn'],
                        required=True, default='graphsage')
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument("--fanout", type=str, required=True, help="fanout numbers", default='20,20')
    parser.add_argument('--batch_size', type=int, required=True, default=1)
    parser.add_argument('--GPU', nargs='+', type=int, required=True)
    parser.add_argument('--num_workers_per_gpu', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--out_path', type=str, required=True, help="Absolute path for saving model parameters")
    args = parser.parse_args()

    # parse arguments
    BASE_PATH = args.data_path
    MODEL_CHOICE = args.gnn_model
    HID_DIM = args.hidden_dim
    N_LAYERS = args.n_layers
    FANOUTS = [int(i) for i in args.fanout.split(',')]
    BATCH_SIZE = args.batch_size
    GPUS = args.GPU
    WORKERS = args.num_workers_per_gpu
    EPOCHS = args.epochs
    OUT_PATH = args.out_path

    # output arguments for logging
    print('Data path: {}'.format(BASE_PATH))
    print('Used algorithm: {}'.format(MODEL_CHOICE))
    print('Hidden dimensions: {}'.format(HID_DIM))
    print('number of hidden layers: {}'.format(N_LAYERS))
    print('Fanout list: {}'.format(FANOUTS))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('GPU list: {}'.format(GPUS))
    print('Number of workers per GPU: {}'.format(WORKERS))
    print('Max number of epochs: {}'.format(EPOCHS))
    print('Output path: {}'.format(OUT_PATH))

    # Retrieve preprocessed data and add reverse edge and self-loop
    graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph(BASE_PATH)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    graph.create_formats_()

    # call train with CPU, one GPU, or multiple GPUs
    if GPUS[0] < 0:
        cpu_device = th.device('cpu')
        cpu_train(graph_data=(graph, labels, train_nid, val_nid, test_nid, node_feat),
                  gnn_model=MODEL_CHOICE,
                  n_layers=N_LAYERS,
                  hidden_dim=HID_DIM,
                  n_classes=23,
                  fanouts=FANOUTS,
                  batch_size=BATCH_SIZE,
                  num_workers=WORKERS,
                  device=cpu_device,
                  epochs=EPOCHS,
                  out_path=OUT_PATH)
    else:
        n_gpus = len(GPUS)

        if n_gpus == 1:
            gpu_train(0, n_gpus, GPUS,
                      graph_data=(graph, labels, train_nid, val_nid, test_nid, node_feat),
                      gnn_model=MODEL_CHOICE, hidden_dim=HID_DIM, n_layers=N_LAYERS, n_classes=23,
                      fanouts=FANOUTS, batch_size=BATCH_SIZE, num_workers=WORKERS, epochs=EPOCHS,
                      message_queue=None, output_folder=OUT_PATH)
        else:
            message_queue = mp.Queue()
            procs = []
            for proc_id in range(n_gpus):
                p = mp.Process(target=gpu_train,
                               args=(proc_id, n_gpus, GPUS,
                                     (graph, labels, train_nid, val_nid, test_nid, node_feat),
                                     MODEL_CHOICE, HID_DIM, N_LAYERS, 23,
                                     FANOUTS, BATCH_SIZE, WORKERS, EPOCHS,
                                     message_queue, OUT_PATH))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()