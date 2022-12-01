import time as tm
from functools import reduce

import torch
import numpy as np
import mlflow as mlf
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from digg.generator.loading import create, split_data, GraphSequenceSampler
from digg.generator.evaluation import bootstrap_eval
from digg.generator.model import PlainGRU
from digg.generator.mlf_utils import (
    mlf_save_text,
    mlf_get_run,
    mlf_log_from_omegaconf_dict,
)
from digg.generator.train_utils import (
    checkpoint,
    save_best,
    binary_cross_entropy_weight,
)


def train(cfg, run_dir):
    seed = cfg.training.seed
    mlf_kwargs = cfg.mlflow
    data_kwargs = cfg.data
    model_kwargs = cfg.model
    train_kwargs = cfg.training
    rng = np.random.default_rng(seed)
    mlf_run = mlf_get_run(run_dir=run_dir, **mlf_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with mlf.start_run(run_id=mlf_run.info.run_id):
        mlf_log_from_omegaconf_dict(cfg)
        graphs = create(
            data_kwargs.graph_type,
            data_kwargs.caida_source_path,
            data_kwargs.data_size,
            data_kwargs.min_num_node,
            data_kwargs.max_num_node,
            data_kwargs.check_size,
        )
        train_graphs, val_graphs, test_graphs = split_data(
            graphs,
            rng,
            with_val=True,
            graph_type=data_kwargs.graph_type,
            inplace=data_kwargs.inplace,
        )
        for stage, s_graphs in [
            ("train", train_graphs),
            ("validation", val_graphs),
            ("test", test_graphs),
        ]:
            mlf_save_text(
                f"{stage}_graphs.csv",
                "caida_graphs",
                reduce(lambda a, b: f"{a}\n{b}", s_graphs._list),
            )

        dataset = GraphSequenceSampler(
            train_graphs,
            max_prev_node=data_kwargs.max_prev_node,
            max_num_node=data_kwargs.max_num_node,
        )
        sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
            [1.0 / len(dataset) for _ in range(len(dataset))],
            num_samples=data_kwargs.data_size,
            replacement=True,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=data_kwargs.batch_size,
            num_workers=data_kwargs.num_workers,
            sampler=sample_strategy,
        )
        rnn = PlainGRU(
            input_size=data_kwargs.max_prev_node,
            embedding_size=model_kwargs.embedding_size_rnn,
            hidden_size=model_kwargs.hidden_size_rnn,
            num_layers=model_kwargs.num_layer,
            has_input=True,
            has_output=True,
            output_size=model_kwargs.hidden_size_rnn_output,
            device=device,
        ).to(device)
        output = PlainGRU(
            input_size=1,
            embedding_size=model_kwargs.embedding_size_rnn_output,
            hidden_size=model_kwargs.hidden_size_rnn_output,
            num_layers=model_kwargs.num_layer,
            has_input=True,
            has_output=True,
            output_size=1,
            device=device,
        ).to(device)
        run_training(
            data_loader,
            rnn,
            output,
            val_graphs=val_graphs,
            rng=rng,
            device=device,
            num_layer=model_kwargs.num_layer,
            min_num_node=data_kwargs.min_num_node,
            max_num_node=data_kwargs.max_num_node,
            max_prev_node=data_kwargs.max_prev_node,
            train_kwargs=train_kwargs,
        )


def run_training(
    data_loader,
    rnn,
    output,
    val_graphs,
    rng,
    device,
    num_layer,
    min_num_node,
    max_num_node,
    max_prev_node,
    train_kwargs,
):
    idx = 0
    epoch = 1
    best_mmd_values = {}
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=train_kwargs.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=train_kwargs.lr)
    scheduler_rnn = MultiStepLR(
        optimizer_rnn, milestones=train_kwargs.milestones, gamma=train_kwargs.lr_rate
    )
    scheduler_output = MultiStepLR(
        optimizer_output, milestones=train_kwargs.milestones, gamma=train_kwargs.lr_rate
    )
    for m in train_kwargs.metrics:
        best_mmd_values[m] = 0
    while epoch <= train_kwargs.num_epochs:
        time_start = tm.time()
        avg_loss, raw_signatures = epoch_training(
            epoch,
            data_loader,
            rnn,
            output,
            num_layer,
            optimizer_rnn,
            optimizer_output,
            scheduler_rnn,
            scheduler_output,
            device,
        )
        time_end = tm.time()
        mlf.log_metric("loss", avg_loss.item(), step=epoch)
        mlf.log_metric("epoch_time", time_end - time_start, step=epoch)
        if (
            epoch % train_kwargs.epochs_test == 0
            and epoch >= train_kwargs.epochs_test_start
        ):
            pred_graphs = checkpoint(
                rnn,
                output,
                epoch,
                idx,
                min_num_node,
                max_num_node,
                max_prev_node,
                num_layer,
                device,
                train_kwargs.test_batch_size,
                train_kwargs.test_total_size,
                raw_signatures,
            )
            mean_values, _, _ = bootstrap_eval(
                val_graphs,
                pred_graphs,
                rng,
                train_kwargs.metrics,
                n_samples=train_kwargs.n_bootstrap_samples,
            )
            for m in train_kwargs.metrics:
                if mean_values[m] > best_mmd_values[m]:
                    save_best(
                        rnn,
                        output,
                        pred_graphs,
                        mean_values[m],
                        f"mmd_{m}",
                        epoch,
                        raw_signatures,
                    )
                mlf.log_metric(f"mmd_{m}", mean_values[m], step=epoch)
            idx = idx + 1 if (idx + 1) % train_kwargs.n_checkpoints != 0 else 0
        epoch += 1


def epoch_training(
    epoch,
    data_loader,
    rnn,
    output,
    num_layer,
    optimizer_rnn,
    optimizer_output,
    scheduler_rnn,
    scheduler_output,
    device,
):
    loss_sum = 0
    rnn.train()
    output.train()
    bar = tqdm(total=len(data_loader), desc=f"Epoch {epoch}: Batches")
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data["x"].float()
        y_unsorted = data["y"].float()
        y_len_unsorted = data["len"]
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        # a smart use of pytorch builtin function:
        # pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        output_x = torch.cat(
            (torch.ones(y_reshape.size(0), 1, 1), y_reshape[:, 0:-1, 0:1]), dim=1
        )
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])  # count how many y_len is above i
            output_y_len.extend(
                [min(i, y.size(2))] * count_temp
            )  # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(device)
        y = Variable(y).to(device)
        output_x = Variable(output_x).to(device)
        output_y = Variable(output_y).to(device)
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())

        _h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(
            _h, y_len, batch_first=True
        ).data  # get packed hidden vector
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(num_layer - 1, h.size(0), h.size(1))).to(
            device
        )
        output.hidden = torch.cat(
            (h.view(1, h.size(0), h.size(1)), hidden_null), dim=0
        )  # num_layer, batch_size, hidden_size
        _y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = torch.sigmoid(_y_pred)

        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.data * feature_dim
        bar.update()
    bar.close()
    return loss_sum / (batch_idx + 1), {
        "rnn": [x.cpu().detach().numpy(), _h.cpu().detach().numpy()],
        "output": [output_x.cpu().detach().numpy(), _y_pred.cpu().detach().numpy()],
    }
