import argparse
import gc
import hashlib
import math
import os
import random
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from torch.utils.data import DataLoader

from dataset_loader import Pdbbindset, collate as base_collate





from distributed_utils import (
    GlobalBatchDistributedSampler,
    barrier,
    broadcast_bool,
    broadcast_object,
    cleanup_distributed,
    ddp_mean_loss,
    distributed_gcl_loss,
    gather_variable_1d,
    init_distributed,
    load_flexible_state_dict,
    reduce_sum,
    seed_worker,
)
from hetero_graph import (
    affinity_edge_config,
    HeteroGraphBuilder,
    MMCLKinWithHeteroView,
    NormalizedGatedHeteroGraphEncoder,
    build_support_graph,
    load_or_build_frozen_entity_cache,
    parse_sample_ids,
    records_for_pairs,
    validate_no_leakage,
)
from metric import regression_metrics
from model import MMCLKin


warnings.filterwarnings("ignore")
Pair = Tuple[int, int]
Record = Tuple[int, int, float, str]


class Logger:
    def __init__(self, file_name: str, stream) -> None:
        self.terminal = stream
        self.log = open(file_name, "a", buffering=1)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dataset", type=str, default="3dkdavis")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/dell/chh/3dkdavis/3dkdavis_gra_seq_pts",
    )
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--label", default="new_kinase", type=str)
    parser.add_argument("--EPOCHS", default=150, type=int)
    parser.add_argument("--early_stop_patience", default=50, type=int)
    parser.add_argument("--model_save_path", default="3dkdavis_kinase-residual", type=str)
    parser.add_argument(
        "--output_root", default="/home/dell/chh/output/3dkdavis", type=str
    )
    parser.add_argument("--node_in_dim", default=6, type=int)
    parser.add_argument("--node_h_dim", default=6, type=int)
    parser.add_argument("--edge_in_dim", default=32, type=int)
    parser.add_argument("--edge_h_dim", default=32, type=int)
    parser.add_argument("--model", default="MMCLKin", type=str)
    parser.add_argument("--lr_reduce_patience", default=50, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_heads", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-8, type=float)
    parser.add_argument(
        "--scheduler", default="CosineAnnealingWarmRestarts", type=str
    )
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--loss", default="mse", type=str)
    parser.add_argument("--lstm_dropout", default=0.2, type=float)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--alpha", default=0.2, type=float)

    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--global_batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_hetero_graph", action="store_true", default=True)
    parser.add_argument("--graph_query_count", default=200, type=int)
    parser.add_argument(
        "--graph_query_mode",
        default="new_kinase_entity",
        choices=["pair", "new_kinase_entity"],
    )
    parser.add_argument("--graph_query_kinase_count", default=8, type=int)
    parser.add_argument("--graph_query_rounds", default=4, type=int)
    parser.add_argument("--lambda_graph", default=0.1, type=float)
    parser.add_argument("--graph_hidden_dim", default=256, type=int)
    parser.add_argument("--graph_num_layers", default=2, type=int)
    parser.add_argument(
        "--graph_node_feature_mode", default="raw", choices=["raw", "base_top10"]
    )
    parser.add_argument(
        "--graph_relation_mode", default="legacy", choices=["legacy", "normalized_gate"]
    )
    parser.add_argument(
        "--graph_affinity_mode", default="weighted", choices=["weighted", "edge_attr"]
    )
    parser.add_argument(
        "--graph_affinity_message", default="additive", choices=["additive", "nonlinear"]
    )
    parser.add_argument("--graph_affinity_source_beta", default=0.0, type=float)
    parser.add_argument("--graph_low_over_lambda", default=0.0, type=float)
    parser.add_argument("--drug_topk", default=5, type=int)
    parser.add_argument("--kinase_topk", default=10, type=int)
    parser.add_argument("--drug_sim_threshold", default=None, type=float)
    parser.add_argument("--kinase_sim_threshold", default=None, type=float)
    parser.add_argument(
        "--strict_inductive_graph",
        dest="strict_inductive_graph",
        action="store_true",
    )
    parser.add_argument(
        "--no_strict_inductive_graph",
        dest="strict_inductive_graph",
        action="store_false",
    )
    parser.add_argument(
        "--graph_update_base", dest="graph_update_base", action="store_true"
    )
    parser.add_argument(
        "--no_graph_update_base", dest="graph_update_base", action="store_false"
    )
    parser.set_defaults(strict_inductive_graph=True, graph_update_base=True)
    parser.add_argument(
        "--fusion_mode",
        default="direct_concat",
        choices=["direct_concat", "residual"],
    )
    parser.add_argument("--graph_max_delta", default=1.0, type=float)
    parser.add_argument("--graph_only", action="store_true")
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--graph_lr", default=1e-4, type=float)
    parser.add_argument("--graph_weight_decay", default=1e-6, type=float)
    parser.add_argument("--graph_gradient_clip", default=1.0, type=float)
    parser.add_argument("--graph_global_batch_size", default=64, type=int)
    parser.add_argument("--graph_monitor_kinase_count", default=4, type=int)
    parser.add_argument("--graph_monitor_seed", default=4321, type=int)
    parser.add_argument("--graph_early_stop_patience", default=15, type=int)
    parser.add_argument("--graph_min_delta", default=1e-4, type=float)
    parser.add_argument(
        "--cache_dir", default="/home/dell/chh/MMCLKin-main/cache", type=str
    )
    parser.add_argument("--rebuild_graph_cache", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--gradient_clip", default=0.0, type=float)
    parser.add_argument("--log_interval", default=50, type=int)
    return parser


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def resolve_csv_path(args) -> str:
    if args.csv_path:
        return args.csv_path
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(args.data_path)),
            "new_{}_overall.csv".format(args.dataset),
        ),
        "/home/dell/chh/{}/new_{}_overall.csv".format(args.dataset, args.dataset),
        os.path.join(
            os.path.dirname(__file__),
            "datasets",
            "3DKDavis",
            "new_3dkdavis_overall.csv",
        ),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("cannot locate the 3DKDavis CSV")


def process_data(
    path: str, datacsv_path: str, label: str
) -> Tuple[List[str], List[str]]:
    datasets = sorted(
        filename for filename in os.listdir(path) if filename.endswith(".pt")
    )
    dataframe = pd.read_csv(datacsv_path)
    protein_ids = list(set(int(item) for item in dataframe["protein_id"]))
    drug_ids = list(set(int(item) for item in dataframe["drug_id"]))
    split_rng = random.Random(1234)
    train_dataset: List[str] = []
    valid_dataset: List[str] = []

    if label == "new_drug":
        split_rng.shuffle(drug_ids)
        split_index = int(len(drug_ids) * 0.815)
        train_entities = set(drug_ids[:split_index])
        valid_entities = set(drug_ids[split_index:])
        for filename in datasets:
            drug_id, _ = parse_sample_ids(filename)
            if drug_id in train_entities:
                train_dataset.append(filename)
            elif drug_id in valid_entities:
                valid_dataset.append(filename)
    elif label == "new_kinase":
        split_rng.shuffle(protein_ids)
        split_index = int(len(protein_ids) * 0.81)
        train_entities = set(protein_ids[:split_index])
        valid_entities = set(protein_ids[split_index:])
        for filename in datasets:
            _, protein_id = parse_sample_ids(filename)
            if protein_id in train_entities:
                train_dataset.append(filename)
            elif protein_id in valid_entities:
                valid_dataset.append(filename)
    elif label == "both_new":
        split_rng.shuffle(drug_ids)
        split_rng.shuffle(protein_ids)
        protein_valid = set(protein_ids[: int(len(protein_ids) * 0.34)])
        protein_train = set(protein_ids[int(len(protein_ids) * 0.34) :])
        drug_valid = set(drug_ids[: int(len(drug_ids) * 0.338)])
        drug_train = set(drug_ids[int(len(drug_ids) * 0.338) :])
        for filename in datasets:
            drug_id, protein_id = parse_sample_ids(filename)
            if drug_id in drug_train and protein_id in protein_train:
                train_dataset.append(filename)
            elif drug_id in drug_valid and protein_id in protein_valid:
                valid_dataset.append(filename)
    else:
        raise ValueError("unsupported split label: {}".format(label))

    if not train_dataset or not valid_dataset:
        raise RuntimeError("empty train or validation split")
    return train_dataset, valid_dataset


def collate(sample_paths: Sequence[str]):
    x_feats = base_collate(sample_paths)
    pairs = [parse_sample_ids(path) for path in sample_paths]
    x_feats.drug_id = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    x_feats.protein_id = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    return x_feats


def attach_node_indices(x_feats, cache: Dict) -> None:
    drug_map = cache["drug_id_to_index"]
    kinase_map = cache["kinase_id_to_index"]
    drug_ids = [int(item) for item in x_feats.drug_id.tolist()]
    kinase_ids = [int(item) for item in x_feats.protein_id.tolist()]
    missing_drugs = [item for item in drug_ids if item not in drug_map]
    missing_kinases = [item for item in kinase_ids if item not in kinase_map]
    if missing_drugs or missing_kinases:
        raise KeyError(
            "missing graph IDs drugs={} kinases={}".format(
                missing_drugs[:5], missing_kinases[:5]
            )
        )
    x_feats.drug_node_index = torch.tensor(
        [drug_map[item] for item in drug_ids], dtype=torch.long
    )
    x_feats.kinase_node_index = torch.tensor(
        [kinase_map[item] for item in kinase_ids], dtype=torch.long
    )


def make_dataloader(
    filenames: Sequence[str],
    data_path: str,
    global_batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int,
) -> Tuple[DataLoader, GlobalBatchDistributedSampler]:
    dataset = Pdbbindset(list(filenames), data_path)
    sampler = GlobalBatchDistributedSampler(
        len(dataset),
        global_batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataloader, sampler


def regression_loss_sum(
    prediction: torch.Tensor, target: torch.Tensor, loss_name: str
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target, reduction="sum")
    if loss_name == "mae":
        return F.l1_loss(prediction, target, reduction="sum")
    if loss_name in ("huber", "smae"):
        return F.smooth_l1_loss(prediction, target, reduction="sum")
    raise ValueError(
        "DDP regression scaling supports mse, mae, huber, and smae; got {}".format(
            loss_name
        )
    )


def graph_regression_loss_sum(prediction, target, loss_name, low_over_lambda=0.0):
    regression_sum = regression_loss_sum(prediction, target, loss_name)
    if low_over_lambda == 0:
        return regression_sum, regression_sum, None
    if not math.isfinite(low_over_lambda) or low_over_lambda < 0 or loss_name != "mse":
        raise ValueError("low overprediction loss requires MSE and finite nonnegative lambda")
    # Query labels are used only as training supervision, never as graph inputs.
    low_over_sum = (F.relu(prediction - target).square() * (target < 6).to(prediction.dtype)).sum()
    return regression_sum + low_over_lambda * low_over_sum, regression_sum, low_over_sum


def low_over_loss_config(weight):
    return {"version": 1, "lambda": float(weight), "threshold": 6.0,
            "penalty": "squared_positive_prediction_error", "denominator": "all_query_samples"}


def build_optimizer(model, args, params):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optimizer == "Adam":
        return Adam(
            trainable,
            lr=params["learning_rate"],
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "SGD":
        return SGD(
            trainable,
            lr=params["learning_rate"],
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "AdamW":
        return AdamW(
            trainable,
            lr=params["learning_rate"],
            weight_decay=args.weight_decay,
        )
    raise ValueError("unsupported optimizer: {}".format(args.optimizer))


def build_scheduler(optimizer, args, params):
    if args.scheduler == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params["lr_reduce_rate"],
            patience=args.lr_reduce_patience,
            min_lr=1e-6,
        )
    if args.scheduler == "ExponentialLR":
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    if args.scheduler == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2
        )
    raise ValueError("unsupported scheduler: {}".format(args.scheduler))


def compute_base_loss(outputs, target, args, params):
    prediction = outputs[0].reshape(-1)
    local_affinity_sum = regression_loss_sum(prediction, target, args.loss)
    affinity_backward, affinity_global_mean = ddp_mean_loss(
        local_affinity_sum, target.numel()
    )
    gcl_results = [
        distributed_gcl_loss(outputs[1], outputs[2]),
        distributed_gcl_loss(outputs[3], outputs[4]),
        distributed_gcl_loss(outputs[5], outputs[6]),
        distributed_gcl_loss(outputs[7], outputs[8]),
    ]
    gcl_backward = [item[0] for item in gcl_results]
    gcl_global = [item[1] for item in gcl_results]
    total = (
        affinity_backward
        + params["beta1"] * gcl_backward[0]
        + params["beta2"] * gcl_backward[1]
        + params["beta3"] * gcl_backward[2]
        + params["beta4"] * gcl_backward[3]
    )
    global_total = (
        affinity_global_mean
        + params["beta1"] * gcl_global[0]
        + params["beta2"] * gcl_global[1]
        + params["beta3"] * gcl_global[2]
        + params["beta4"] * gcl_global[3]
    )
    return total, affinity_global_mean, gcl_global, global_total


def raw_model(model):
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_state: Dict,
    args,
    params: Dict,
    cache: Optional[Dict],
    graph_optimizer=None,
    monitor_state: Optional[Dict] = None,
) -> None:
    payload = {
        "model": raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_mse": best_state.get("best_fusion_mse", best_state.get("best_mse")),
        "best_pearson": best_state.get(
            "best_fusion_pearson", best_state.get("best_pearson")
        ),
        "best_epoch": best_state.get(
            "best_fusion_pearson_epoch", best_state.get("best_epoch", -1)
        ),
        "best_fusion_mse": best_state.get("best_fusion_mse", 1e9),
        "best_fusion_mse_epoch": best_state.get("best_fusion_mse_epoch", -1),
        "best_fusion_pearson": best_state.get("best_fusion_pearson", -1e9),
        "best_fusion_pearson_epoch": best_state.get("best_fusion_pearson_epoch", -1),
        "best_base_mse": best_state.get("best_base_mse", 1e9),
        "best_base_mse_epoch": best_state.get("best_base_mse_epoch", -1),
        "best_base_pearson": best_state.get("best_base_pearson", -1e9),
        "best_base_pearson_epoch": best_state.get("best_base_pearson_epoch", -1),
        "args": vars(args),
        "params": params,
        "random_seed": args.seed,
        "fusion_mode": args.fusion_mode,
        "graph_query_mode": args.graph_query_mode,
    }
    if graph_optimizer is not None:
        payload["graph_optimizer"] = graph_optimizer.state_dict()
    if getattr(args, "graph_low_over_lambda", 0.0) > 0:
        payload["graph_low_over_config"] = low_over_loss_config(args.graph_low_over_lambda)
    if monitor_state is not None:
        payload["graph_monitor"] = monitor_state
    if cache is not None:
        payload.update(
            {
                "drug_id_to_index": cache["drug_id_to_index"],
                "kinase_id_to_index": cache["kinase_id_to_index"],
                "graph_params": cache["graph_params"],
                "graph_cache_hash": cache["cache_hash"],
                "affinity_normalization": {
                    "min": cache["affinity_min"],
                    "max": cache["affinity_max"],
                    "source": cache["affinity_normalization_source"],
                },
            }
        )
        if cache.get("graph_node_feature_mode") == "base_top10":
            payload["frozen_entity_signature"] = cache["frozen_entity_signature"]
        if cache.get("graph_affinity_mode", "weighted") == "edge_attr":
            payload["graph_affinity_config"] = affinity_edge_config(
                getattr(args, "graph_affinity_message", "additive"),
                getattr(args, "graph_affinity_source_beta", 0.0),
            )
    torch.save(payload, path)


def resume_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    rank: int,
    expected_graph_config: Optional[Dict] = None,
    graph_optimizer=None,
) -> Tuple[int, Dict]:
    checkpoint = torch.load(path, map_location=device)
    if expected_graph_config is not None:
        saved_args = checkpoint.get("args", {})
        for key, default in (("graph_node_feature_mode", "raw"), ("graph_relation_mode", "legacy")):
            if saved_args.get(key, default) != expected_graph_config[key]:
                raise ValueError("resume {} mismatch: checkpoint={} requested={}".format(
                    key, saved_args.get(key, default), expected_graph_config[key]
                ))
        requested_affinity = expected_graph_config.get("graph_affinity_mode", "weighted")
        if saved_args.get("graph_affinity_mode", "weighted") != requested_affinity:
            raise ValueError("resume graph_affinity_mode mismatch")
        requested_message = expected_graph_config.get("graph_affinity_message", "additive")
        if saved_args.get("graph_affinity_message", "additive") != requested_message:
            raise ValueError("resume graph_affinity_message mismatch")
        requested_beta = expected_graph_config.get("graph_affinity_source_beta", 0.0)
        if saved_args.get("graph_affinity_source_beta", 0.0) != requested_beta:
            raise ValueError("resume graph_affinity_source_beta mismatch")
        if requested_affinity == "edge_attr" and checkpoint.get("graph_affinity_config") != affinity_edge_config(requested_message, requested_beta):
            raise ValueError("resume affinity edge configuration mismatch")
        requested_low_over = expected_graph_config.get("graph_low_over_lambda", 0.0)
        if saved_args.get("graph_low_over_lambda", 0.0) != requested_low_over:
            raise ValueError("resume graph_low_over_lambda mismatch")
        if requested_low_over > 0 and checkpoint.get("graph_low_over_config") != low_over_loss_config(requested_low_over):
            raise ValueError("resume low overprediction configuration mismatch")
        if expected_graph_config.get("frozen_entity_signature") is not None:
            if checkpoint.get("frozen_entity_signature") != expected_graph_config["frozen_entity_signature"]:
                raise ValueError("resume frozen Base/entity feature cache mismatch")
    state_dict = checkpoint.get("model", checkpoint)
    new_graph_mode = expected_graph_config is not None and (
        expected_graph_config["graph_node_feature_mode"] != "raw"
        or expected_graph_config["graph_relation_mode"] != "legacy"
    )
    if new_graph_mode:
        raw_model(model).load_state_dict(state_dict, strict=True)
        missing, unexpected = [], []
        if graph_optimizer is not None and checkpoint.get("graph_optimizer") is not None:
            graph_optimizer.load_state_dict(checkpoint["graph_optimizer"])
    else:
        missing, unexpected = load_flexible_state_dict(raw_model(model), state_dict)
    if rank == 0:
        print(
            "resume model: missing_keys={} unexpected_keys={}".format(
                len(missing), len(unexpected)
            )
        )
    optimizer_state = checkpoint.get("optimizer", checkpoint.get("optim"))
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except ValueError as exc:
            print("optimizer state not restored: {}".format(exc))
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    state = {
        "best_mse": float(checkpoint.get("best_mse", checkpoint.get("mse", 1e9))),
        "best_pearson": float(checkpoint.get("best_pearson", checkpoint.get("pearson", -1e9))),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_fusion_mse": float(
            checkpoint.get("best_fusion_mse", checkpoint.get("best_mse", 1e9))
        ),
        "best_fusion_mse_epoch": int(
            checkpoint.get("best_fusion_mse_epoch", checkpoint.get("best_epoch", -1))
        ),
        "best_fusion_pearson": float(
            checkpoint.get("best_fusion_pearson", checkpoint.get("best_pearson", -1e9))
        ),
        "best_fusion_pearson_epoch": int(
            checkpoint.get("best_fusion_pearson_epoch", checkpoint.get("best_epoch", -1))
        ),
        "best_base_mse": float(checkpoint.get("best_base_mse", 1e9)),
        "best_base_mse_epoch": int(checkpoint.get("best_base_mse_epoch", -1)),
        "best_base_pearson": float(checkpoint.get("best_base_pearson", -1e9)),
        "best_base_pearson_epoch": int(checkpoint.get("best_base_pearson_epoch", -1)),
        "early_stop_count": 0,
    }
    return start_epoch, state


def train_base_epoch(
    model,
    dataloader,
    sampler,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    args,
    params,
    rank,
) -> Dict:
    model.train()
    sampler.set_epoch(epoch)
    totals = np.zeros(6, dtype=np.float64)
    steps = 0
    for step, x_feats in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)
        x_feats = x_feats.to(device, non_blocking=True)
        with autocast(enabled=args.use_amp):
            outputs = model(x_feats)
            target = x_feats.y.reshape(-1)
            loss, affinity, gcl_values, global_total = compute_base_loss(
                outputs, target, args, params
            )
        scaler.scale(loss).backward()
        if args.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        if args.scheduler != "ReduceLROnPlateau":
            scheduler.step()

        values = [
            float(global_total.item()),
            float(affinity.item()),
            *[float(item.item()) for item in gcl_values],
        ]
        totals += np.asarray(values)
        steps += 1
        if rank == 0 and step % args.log_interval == 0:
            print(
                "# [{}/{}] base step {}/{} total={:.5f} affinity={:.5f} "
                "gcl=[{:.5f}, {:.5f}, {:.5f}, {:.5f}]".format(
                    epoch,
                    args.EPOCHS,
                    step + 1,
                    len(dataloader),
                    *values,
                )
            )
        if args.dry_run:
            break
    return {
        "total": totals[0] / steps,
        "affinity": totals[1] / steps,
        "gcl": (totals[2:] / steps).tolist(),
        "steps": steps,
    }


def select_query_pairs(cache: Dict, count: int, seed: int, epoch: int) -> Set[Pair]:
    unique_pairs = sorted(
        {(record[0], record[1]) for record in cache["train_records"]}
    )
    query_count = min(count, len(unique_pairs))
    rng = random.Random(seed + epoch)
    return set(rng.sample(unique_pairs, query_count))


def query_hash(query_pairs: Set[Pair]) -> str:
    payload = ";".join("{}:{}".format(*pair) for pair in sorted(query_pairs))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def records_for_kinases(records: Sequence[Record], kinase_ids: Set[int]) -> List[Record]:
    return [record for record in records if record[1] in kinase_ids]


def _cycle_permutation(items: Sequence[int], seed: int, cycle_index: int) -> List[int]:
    shuffled = list(sorted(items))
    rng = random.Random(seed + cycle_index)
    rng.shuffle(shuffled)
    return shuffled


def select_pseudo_cold_kinase_rounds(
    train_kinase_indices: Sequence[int],
    count: int,
    rounds: int,
    seed: int,
    epoch: int,
    excluded_kinases: Optional[Set[int]] = None,
) -> List[List[int]]:
    candidates = sorted(set(train_kinase_indices) - set(excluded_kinases or set()))
    if not candidates:
        raise ValueError("no train kinases available for pseudo-cold selection")
    per_epoch_count = int(count) * int(rounds)
    if per_epoch_count <= 0:
        raise ValueError("count * rounds must be positive")
    if len(candidates) < count:
        raise ValueError("not enough train kinases for one pseudo-cold round")

    selected: List[int] = []
    position = epoch * per_epoch_count
    while len(selected) < per_epoch_count:
        cycle_index, offset = divmod(position, len(candidates))
        permutation = _cycle_permutation(candidates, seed, cycle_index)
        selected_set = set(selected)
        available = permutation[offset:]
        if len(candidates) >= per_epoch_count:
            available = [item for item in available if item not in selected_set]
        if not available:
            position = (cycle_index + 1) * len(candidates)
            continue
        needed = per_epoch_count - len(selected)
        selected.extend(available[:needed])
        position += len(available[:needed])

    return [
        selected[start : start + count]
        for start in range(0, per_epoch_count, count)
    ]


def fixed_monitor_kinases(
    train_kinase_indices: Sequence[int], count: int, seed: int
) -> List[int]:
    candidates = sorted(set(train_kinase_indices))
    if count <= 0:
        return []
    if len(candidates) < count:
        raise ValueError("not enough train kinases for monitor")
    return _cycle_permutation(candidates, seed, 0)[:count]


def graph_train_summary(round_stats: Sequence[Dict], kinase_rounds: Sequence[Sequence[int]]) -> Dict:
    total_query_count = sum(int(item["query_record_count"]) for item in round_stats)
    if total_query_count <= 0:
        raise ValueError("graph query count must be positive")
    weighted_loss = sum(
        float(item["loss"]) * int(item["query_record_count"]) for item in round_stats
    ) / total_query_count
    support_mean = sum(float(item["support_edge_count"]) for item in round_stats) / len(round_stats)
    unique_kinases = sorted({kinase for group in kinase_rounds for kinase in group})
    return {
        "loss": weighted_loss,
        "rounds": len(round_stats),
        "query_record_count": total_query_count,
        "unique_pseudo_cold_kinases": len(unique_kinases),
        "support_edge_count_mean": support_mean,
        "round_stats": list(round_stats),
    }


def set_graph_only_train_mode(model) -> None:
    wrapper = raw_model(model)
    wrapper.base_model.eval()
    wrapper.hetero_encoder.train()
    wrapper.fusion_head.train()


def freeze_base_model(model) -> None:
    wrapper = raw_model(model)
    for parameter in wrapper.base_model.parameters():
        parameter.requires_grad = False
    wrapper.base_model.eval()


def build_graph_optimizer(model, graph_lr: float, graph_weight_decay: float):
    wrapper = raw_model(model)
    base_param_ids = {id(parameter) for parameter in wrapper.base_model.parameters()}
    graph_parameters = [
        parameter
        for parameter in list(wrapper.hetero_encoder.parameters())
        + list(wrapper.fusion_head.parameters())
        if parameter.requires_grad
    ]
    graph_param_ids = {id(parameter) for parameter in graph_parameters}
    if not graph_param_ids:
        raise ValueError("graph optimizer has no trainable parameters")
    if not graph_param_ids.isdisjoint(base_param_ids):
        raise AssertionError("graph optimizer contains base parameters")
    optimizer = AdamW(
        graph_parameters,
        lr=graph_lr,
        weight_decay=graph_weight_decay,
    )
    return optimizer, graph_parameters


def _has_finite_nonzero_grad(parameters: Sequence[torch.nn.Parameter]) -> bool:
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            if torch.isfinite(grad).all() and float(grad.abs().sum().item()) > 0:
                return True
    return False


def _all_grads_none(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(parameter.grad is None for parameter in parameters)


def assert_graph_only_gradients(model) -> None:
    wrapper = raw_model(model)
    if not all(not parameter.requires_grad for parameter in wrapper.base_model.parameters()):
        raise AssertionError("base parameters must be frozen in graph_only")
    if not _all_grads_none(list(wrapper.base_model.parameters())):
        raise AssertionError("base gradients must remain None in graph_only")
    graph_params = list(wrapper.hetero_encoder.parameters()) + list(wrapper.fusion_head.parameters())
    if not _has_finite_nonzero_grad(graph_params):
        raise AssertionError("graph parameters did not receive finite nonzero gradients")


def load_base_only_from_checkpoint(
    base_model,
    checkpoint_path: str,
    device: torch.device,
    min_loaded: int = 10,
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint.get("model", checkpoint)
    target_state = base_model.state_dict()
    loaded = {}
    unexpected = []
    mismatched = []
    prefixes = ("module.base_model.", "base_model.", "module.", "")

    for key, value in source_state.items():
        matched_name = None
        for prefix in prefixes:
            if prefix and not key.startswith(prefix):
                continue
            candidate = key[len(prefix) :] if prefix else key
            if candidate in target_state:
                matched_name = candidate
                break
        if matched_name is None:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[matched_name].shape):
            mismatched.append(key)
            continue
        loaded[matched_name] = value.detach().clone()

    target_state.update(loaded)
    base_model.load_state_dict(target_state)
    missing = sorted(set(target_state.keys()) - set(loaded.keys()))
    report = {
        "loaded_key_count": len(loaded),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": mismatched,
    }
    if len(loaded) < min_loaded:
        raise RuntimeError(
            "too few base keys loaded from {}: {}".format(checkpoint_path, report)
        )
    return report


def train_graph_epoch(
    model,
    cache,
    query_pairs,
    optimizer,
    scaler,
    device,
    epoch,
    args,
    rank,
    world_size,
) -> Dict:
    support_graph, support_pairs = build_support_graph(cache, query_pairs)
    valid_pairs = {(record[0], record[1]) for record in cache["valid_records"]}
    if not support_pairs.isdisjoint(query_pairs):
        raise AssertionError("query/support leakage")
    if not support_pairs.isdisjoint(valid_pairs):
        raise AssertionError("validation/support leakage")
    support_graph = support_graph.to(device)
    query_records = records_for_pairs(cache["train_records"], query_pairs)
    query_files = [record[3] for record in query_records]
    dataloader, sampler = make_dataloader(
        query_files,
        args.data_path,
        args.graph_global_batch_size,
        rank,
        world_size,
        args.seed + 100000,
        True,
        args.num_workers,
    )
    sampler.set_epoch(epoch)
    model.train()
    graph_losses = 0.0
    steps = 0
    for step, x_feats in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)
        attach_node_indices(x_feats, cache)
        x_feats = x_feats.to(device, non_blocking=True)
        with autocast(enabled=args.use_amp):
            outputs = model(
                x_feats,
                support_graph,
                True,
                args.graph_update_base,
            )
            fusion_prediction = outputs[1].reshape(-1)
            target = x_feats.y.reshape(-1)
            local_sum = regression_loss_sum(fusion_prediction, target, args.loss)
            graph_backward, graph_global_mean = ddp_mean_loss(
                local_sum, target.numel()
            )
            loss = args.lambda_graph * graph_backward
        scaler.scale(loss).backward()
        if args.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        graph_losses += float(graph_global_mean.item())
        steps += 1
        if rank == 0 and step % args.log_interval == 0:
            print(
                "# [{}/{}] graph step {}/{} graph_loss={:.5f}".format(
                    epoch,
                    args.EPOCHS,
                    step + 1,
                    len(dataloader),
                    graph_global_mean.item(),
                )
            )
        if args.dry_run:
            break
    return {
        "loss": graph_losses / steps,
        "steps": steps,
        "query_count": len(query_pairs),
        "query_record_count": len(query_records),
        "support_edge_count": support_graph[
            ("drug", "binds", "kinase")
        ].edge_index.size(1),
    }


def _record_label_stats(records: Sequence[Record]) -> Dict:
    labels = torch.tensor([record[2] for record in records], dtype=torch.float32)
    return {
        "label_mean": float(labels.mean().item()),
        "label_std": float(labels.std(unbiased=False).item()) if labels.numel() > 1 else 0.0,
        "label_min": float(labels.min().item()),
        "label_max": float(labels.max().item()),
    }


def train_graph_round(
    model,
    cache,
    query_records: Sequence[Record],
    support_graph,
    optimizer,
    scaler,
    device,
    epoch,
    round_idx,
    args,
    rank,
    world_size,
    graph_parameters: Optional[Sequence[torch.nn.Parameter]] = None,
) -> Dict:
    query_files = [record[3] for record in query_records]
    dataloader, sampler = make_dataloader(
        query_files,
        args.data_path,
        args.graph_global_batch_size,
        rank,
        world_size,
        args.seed + 100000 + round_idx,
        True,
        args.num_workers,
    )
    sampler.set_epoch(epoch)
    support_graph = support_graph.to(device)
    if args.graph_only:
        set_graph_only_train_mode(model)
    else:
        model.train()
    weighted_loss = 0.0
    regression_total = 0.0
    low_over_total = 0.0
    seen = 0.0
    steps = 0
    for step, x_feats in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)
        attach_node_indices(x_feats, cache)
        x_feats = x_feats.to(device, non_blocking=True)
        with autocast(enabled=args.use_amp):
            outputs = model(
                x_feats,
                support_graph,
                True,
                False if args.graph_only else args.graph_update_base,
            )
            fusion_prediction = outputs[1].reshape(-1)
            target = x_feats.y.reshape(-1)
            local_sum, regression_sum, low_over_sum = graph_regression_loss_sum(
                fusion_prediction, target, args.loss, getattr(args, "graph_low_over_lambda", 0.0)
            )
            graph_backward, graph_global_mean = ddp_mean_loss(
                local_sum, target.numel()
            )
            loss = graph_backward if args.graph_only else args.lambda_graph * graph_backward
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError("non-finite graph loss")
        scaler.scale(loss).backward()
        if args.graph_only:
            assert_graph_only_gradients(model)
            if args.graph_gradient_clip > 0 and graph_parameters is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    graph_parameters, args.graph_gradient_clip
                )
        elif args.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        global_count = reduce_sum(
            torch.tensor(float(target.numel()), device=device)
        ).item()
        weighted_loss += float(graph_global_mean.item()) * float(global_count)
        if low_over_sum is not None:
            components = reduce_sum(torch.stack((regression_sum.detach(), low_over_sum.detach())))
            regression_total += float(components[0].item())
            low_over_total += float(components[1].item())
        seen += float(global_count)
        steps += 1
        if rank == 0 and step % args.log_interval == 0:
            print(
                "# [{}/{}] graph round {} step {}/{} graph_loss={:.5f}".format(
                    epoch,
                    args.EPOCHS,
                    round_idx + 1,
                    step + 1,
                    len(dataloader),
                    graph_global_mean.item(),
                )
            )
        if args.dry_run:
            break
    result = {
        "loss": weighted_loss / seen,
        "steps": steps,
        "query_record_count": len(query_records),
    }
    if getattr(args, "graph_low_over_lambda", 0.0) > 0:
        result.update(regression_mse=regression_total / seen, low_over_mean=low_over_total / seen)
        if rank == 0:
            print("graph_round_loss_components={}".format(result))
    return result


def train_entity_graph_epoch(
    model,
    cache,
    kinase_rounds: Sequence[Sequence[int]],
    optimizer,
    scaler,
    device,
    epoch,
    args,
    rank,
    world_size,
    graph_parameters: Optional[Sequence[torch.nn.Parameter]] = None,
) -> Dict:
    round_stats = []
    for round_idx, pseudo_cold_kinases in enumerate(kinase_rounds):
        pseudo_set = set(int(item) for item in pseudo_cold_kinases)
        query_records = records_for_kinases(cache["train_records"], pseudo_set)
        if not query_records:
            raise RuntimeError("pseudo-cold kinase round has no query records")
        query_pairs = {(record[0], record[1]) for record in query_records}
        support_graph, support_pairs, support_stats = build_support_graph(
            cache,
            excluded_pairs=query_pairs,
            excluded_kinases=pseudo_set,
            simulate_new_kinase=True,
            return_stats=True,
        )
        if not support_pairs.isdisjoint(query_pairs):
            raise AssertionError("query labels entered support graph")
        round_stat = {
            "round": round_idx + 1,
            "pseudo_cold_kinases": sorted(pseudo_set),
            "query_record_count": len(query_records),
            "query_unique_drugs": len({record[0] for record in query_records}),
            "query_hash": query_hash(query_pairs),
            "support_edge_count": support_stats["support_edge_count"],
            "incoming_similarity_edges": support_stats[
                "pseudo_cold_incoming_similarity_edges"
            ],
            "support_affinity_min": support_stats["support_affinity_min"],
            "support_affinity_max": support_stats["support_affinity_max"],
        }
        round_stat.update(_record_label_stats(query_records))
        if rank == 0:
            print("graph_round={}".format(round_stat))
            if support_stats["pseudo_cold_no_incoming_kinases"]:
                print(
                    "graph_round_warning no_incoming_similarity_kinases={}".format(
                        support_stats["pseudo_cold_no_incoming_kinases"]
                    )
                )
        train_stat = train_graph_round(
            model,
            cache,
            query_records,
            support_graph,
            optimizer,
            scaler,
            device,
            epoch,
            round_idx,
            args,
            rank,
            world_size,
            graph_parameters,
        )
        round_stat.update(train_stat)
        round_stats.append(round_stat)
    return graph_train_summary(round_stats, kinase_rounds)


@torch.no_grad()
def evaluate_fixed_graph_monitor(
    model,
    cache,
    monitor_kinases: Sequence[int],
    device,
    epoch,
    args,
    rank,
    world_size,
) -> Dict:
    monitor_set = set(int(item) for item in monitor_kinases)
    if not monitor_set:
        return {}
    query_records = records_for_kinases(cache["train_records"], monitor_set)
    query_pairs = {(record[0], record[1]) for record in query_records}
    support_graph, _, support_stats = build_support_graph(
        cache,
        excluded_pairs=query_pairs,
        excluded_kinases=monitor_set,
        simulate_new_kinase=True,
        return_stats=True,
    )
    dataloader, sampler = make_dataloader(
        [record[3] for record in query_records],
        args.data_path,
        args.graph_global_batch_size,
        rank,
        world_size,
        args.graph_monitor_seed,
        False,
        args.num_workers,
    )
    sampler.set_epoch(0)
    model.eval()
    support_graph = support_graph.to(device)
    local_true: List[torch.Tensor] = []
    local_base: List[torch.Tensor] = []
    local_fusion: List[torch.Tensor] = []
    for x_feats in dataloader:
        attach_node_indices(x_feats, cache)
        x_feats = x_feats.to(device, non_blocking=True)
        outputs = model(x_feats, support_graph, True, False)
        local_true.append(x_feats.y.reshape(-1))
        local_base.append(outputs[0].reshape(-1))
        local_fusion.append(outputs[1].reshape(-1))
        if args.dry_run:
            break
    all_true = gather_variable_1d(torch.cat(local_true))
    all_base = gather_variable_1d(torch.cat(local_base))
    all_fusion = gather_variable_1d(torch.cat(local_fusion))
    metrics = None
    if rank == 0:
        truth_cpu = all_true.float().cpu()
        fusion_cpu = all_fusion.float().cpu()
        fusion_metrics = regression_metrics(truth_cpu, fusion_cpu)
        metrics = {
            "mse": fusion_metrics["mse"],
            "mae": fusion_metrics["mae"],
            "pearson": fusion_metrics["pearson_value"],
            "query_count": int(all_true.numel()),
            "kinases": sorted(monitor_set),
            "hash": query_hash(query_pairs),
            "incoming_similarity_edges": support_stats[
                "pseudo_cold_incoming_similarity_edges"
            ],
        }
    return broadcast_object(metrics, src=0)


def prediction_diagnostics(
    target: torch.Tensor,
    base_prediction: torch.Tensor,
    fusion_prediction: torch.Tensor,
) -> Tuple[Dict, Dict]:
    base_delta = base_prediction - target
    fusion_delta = fusion_prediction - target
    effective_delta = fusion_prediction - base_prediction
    base_diag = {
        "pred_mean": float(base_prediction.mean().item()),
        "pred_std": float(base_prediction.std(unbiased=False).item()),
        "signed_bias": float(base_delta.mean().item()),
    }
    fusion_diag = {
        "pred_mean": float(fusion_prediction.mean().item()),
        "pred_std": float(fusion_prediction.std(unbiased=False).item()),
        "signed_bias": float(fusion_delta.mean().item()),
        "effective_delta_mean": float(effective_delta.mean().item()),
        "effective_delta_std": float(effective_delta.std(unbiased=False).item()),
        "effective_delta_abs_max": float(effective_delta.abs().max().item()),
        "improvement_rate": float(
            (fusion_delta.abs() < base_delta.abs()).float().mean().item()
        ),
    }
    return base_diag, fusion_diag


@torch.no_grad()
def validate(
    model,
    dataloader,
    sampler,
    cache,
    device,
    epoch,
    args,
    params,
    rank,
) -> Dict:
    model.eval()
    sampler.set_epoch(epoch)
    eval_graph = None
    if cache is not None:
        eval_graph, support_pairs = build_support_graph(cache)
        valid_pairs = {(record[0], record[1]) for record in cache["valid_records"]}
        if not support_pairs.isdisjoint(valid_pairs):
            raise AssertionError("validation affinity edge entered eval graph")
        eval_graph = eval_graph.to(device)
        if rank == 0 and isinstance(raw_model(model).hetero_encoder, NormalizedGatedHeteroGraphEncoder):
            _, diagnostics = raw_model(model).hetero_encoder(eval_graph, collect_diagnostics=True)
            print("graph_relation_diagnostics={}".format(diagnostics))

    local_true: List[torch.Tensor] = []
    local_base: List[torch.Tensor] = []
    local_fusion: List[torch.Tensor] = []
    affinity_sum = 0.0
    gcl_sums = np.zeros(4, dtype=np.float64)
    steps = 0
    for step, x_feats in enumerate(dataloader):
        if cache is not None:
            attach_node_indices(x_feats, cache)
        x_feats = x_feats.to(device, non_blocking=True)
        with autocast(enabled=args.use_amp):
            if cache is None:
                base_outputs = model(x_feats)
                base_prediction = base_outputs[0].reshape(-1)
                fusion_prediction = base_prediction
            else:
                graph_outputs = model(
                    x_feats,
                    eval_graph,
                    True,
                    args.graph_update_base,
                )
                base_prediction = graph_outputs[0].reshape(-1)
                fusion_prediction = graph_outputs[1].reshape(-1)
                base_outputs = (
                    graph_outputs[0],
                    *graph_outputs[2:10],
                )
            target = x_feats.y.reshape(-1)
            local_affinity_sum = regression_loss_sum(
                base_prediction, target, args.loss
            )
            _, global_affinity_mean = ddp_mean_loss(
                local_affinity_sum, target.numel()
            )
            gcl_values = [
                distributed_gcl_loss(base_outputs[1], base_outputs[2])[1],
                distributed_gcl_loss(base_outputs[3], base_outputs[4])[1],
                distributed_gcl_loss(base_outputs[5], base_outputs[6])[1],
                distributed_gcl_loss(base_outputs[7], base_outputs[8])[1],
            ]
        affinity_sum += float(global_affinity_mean.item())
        gcl_sums += np.asarray([float(item.item()) for item in gcl_values])
        steps += 1
        local_true.append(target)
        local_base.append(base_prediction)
        local_fusion.append(fusion_prediction)
        if args.dry_run:
            break

    all_true = gather_variable_1d(torch.cat(local_true))
    all_base = gather_variable_1d(torch.cat(local_base))
    all_fusion = gather_variable_1d(torch.cat(local_fusion))
    metrics = None
    if rank == 0:
        truth_cpu = all_true.float().cpu()
        base_cpu = all_base.float().cpu()
        fusion_cpu = all_fusion.float().cpu()
        metrics = {
            "base": regression_metrics(truth_cpu, base_cpu),
            "fusion": regression_metrics(truth_cpu, fusion_cpu),
            "base_affinity_loss": affinity_sum / steps,
            "gcl": (gcl_sums / steps).tolist(),
            "sample_count": int(all_true.numel()),
        }
        base_diag, fusion_diag = prediction_diagnostics(
            truth_cpu, base_cpu, fusion_cpu
        )
        metrics["base_diagnostics"] = base_diag
        metrics["fusion_diagnostics"] = fusion_diag
    return broadcast_object(metrics, src=0)


def print_environment(args, csv_path: str) -> None:
    sample_files = sorted(
        filename for filename in os.listdir(args.data_path) if filename.endswith(".pt")
    )
    sample_path = os.path.join(args.data_path, sample_files[0])
    sample = torch.load(sample_path, map_location="cpu")
    dataframe = pd.read_csv(csv_path)
    print(
        "environment: python={} torch={} pyg={} dgl={} cuda={} cuda_devices={} "
        "distributed={} nccl={}".format(
            sys.version.split()[0],
            torch.__version__,
            torch_geometric.__version__,
            dgl.__version__,
            torch.version.cuda,
            torch.cuda.device_count(),
            torch.distributed.is_available(),
            torch.distributed.is_nccl_available(),
        )
    )
    print(
        "sample={} mol_embedding={} pro_token_repre={} poc_token_repre={} "
        "com_affinity={} mol_atoms_feats={} pro_atoms_feats_s={}".format(
            os.path.basename(sample_path),
            tuple(sample.mol_embedding.shape),
            tuple(sample.pro_token_repre.shape),
            tuple(sample.poc_token_repre.shape),
            tuple(sample.com_affinity.shape),
            tuple(sample.mol_atoms_feats.shape),
            tuple(sample.pro_atoms_feats_s.shape),
        )
    )
    print("csv columns={} rows={}".format(list(dataframe.columns), len(dataframe)))


def main(args, params) -> None:
    device = None
    stdout_logger = None
    stderr_logger = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        affinity_edge_config(args.graph_affinity_message, args.graph_affinity_source_beta)
        if args.graph_affinity_source_beta > 0 and not (
            args.graph_affinity_mode == "edge_attr" and args.graph_relation_mode == "normalized_gate"
        ):
            raise ValueError("affinity source modulation requires edge_attr and normalized_gate")
        if args.graph_affinity_message == "nonlinear" and args.graph_affinity_mode != "edge_attr":
            raise ValueError("nonlinear affinity message requires edge_attr")
        if not math.isfinite(args.graph_low_over_lambda) or args.graph_low_over_lambda < 0:
            raise ValueError("graph_low_over_lambda must be finite and nonnegative")
        if args.graph_low_over_lambda > 0 and not (
            args.graph_only and args.use_hetero_graph and args.strict_inductive_graph
            and args.base_checkpoint and not args.graph_update_base and args.loss == "mse"
            and args.fusion_mode == "residual" and args.graph_query_mode == "new_kinase_entity"
        ):
            raise ValueError("low overprediction loss requires frozen strict graph_only residual MSE entity training")
        if args.graph_affinity_mode == "edge_attr" and not (
            args.graph_only and args.use_hetero_graph and args.strict_inductive_graph
            and args.base_checkpoint and not args.graph_update_base
            and args.fusion_mode == "residual" and args.graph_relation_mode == "normalized_gate"
        ):
            raise ValueError("edge_attr requires frozen residual graph_only and strict normalized_gate")
        if args.graph_node_feature_mode == "base_top10" and not (
            args.graph_only and args.use_hetero_graph and args.strict_inductive_graph
            and args.base_checkpoint
        ):
            raise ValueError("base_top10 requires frozen graph_only, strict_inductive_graph and base_checkpoint")
        device, local_rank, rank, world_size = init_distributed(args)
        setup_seed(args.seed + rank)
        csv_path = resolve_csv_path(args)
        run_name = None
        if rank == 0:
            run_name = "{}_{}_{}".format(args.model, args.dataset, int(time.time()))
        run_name = broadcast_object(run_name, src=0)
        save_dir = os.path.join(
            args.output_root,
            "{}_{}".format(args.model_save_path, args.label),
            run_name,
        )
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            log_path = os.path.join(save_dir, "{}.log".format(run_name))
            stdout_logger = Logger(log_path, original_stdout)
            stderr_logger = Logger(log_path, original_stderr)
            sys.stdout = stdout_logger
            sys.stderr = stderr_logger
            print_environment(args, csv_path)
            print("args={}".format(args))
        barrier()

        train_files, valid_files = process_data(args.data_path, csv_path, args.label)
        cache = None
        if args.use_hetero_graph:
            cache_path = os.path.join(
                args.cache_dir,
                "{}_{}_hetero_cache.pt".format(args.dataset, args.label),
            )
            if rank == 0:
                builder = HeteroGraphBuilder(
                    args.data_path,
                    csv_path,
                    cache_path,
                    args.drug_topk,
                    args.kinase_topk,
                    args.drug_sim_threshold,
                    args.kinase_sim_threshold,
                    args.strict_inductive_graph,
                )
                builder.load_or_build(
                    train_files,
                    valid_files,
                    rebuild=args.rebuild_graph_cache,
                )
            barrier()
            cache = torch.load(cache_path, map_location="cpu")
            validate_no_leakage(cache, args.label)
            cache["graph_affinity_mode"] = args.graph_affinity_mode
            if rank == 0 and args.graph_affinity_mode == "edge_attr":
                print("graph_affinity_config={}".format(affinity_edge_config(
                    args.graph_affinity_message, args.graph_affinity_source_beta)))
            if rank == 0 and args.graph_low_over_lambda > 0:
                print("graph_low_over_config={}".format(low_over_loss_config(args.graph_low_over_lambda)))

        train_loader, train_sampler = make_dataloader(
            train_files,
            args.data_path,
            args.global_batch_size,
            rank,
            world_size,
            args.seed,
            True,
            args.num_workers,
        )
        valid_loader, valid_sampler = make_dataloader(
            valid_files,
            args.data_path,
            args.global_batch_size,
            rank,
            world_size,
            args.seed + 50000,
            False,
            args.num_workers,
        )
        if rank == 0:
            first_global_size = min(args.global_batch_size, len(train_files))
            base, remainder = divmod(first_global_size, world_size)
            partition = [
                base + (1 if item < remainder else 0)
                for item in range(world_size)
            ]
            print(
                "split: train={} valid={} global_batch={} first_local_batches={}".format(
                    len(train_files),
                    len(valid_files),
                    args.global_batch_size,
                    partition,
                )
            )
            if cache is not None:
                print(
                    "graph cache={} hash={} drugs={} kinases={} train_pairs={} "
                    "valid_pairs={} affinity_range=[{:.6f}, {:.6f}]".format(
                        cache_path,
                        cache["cache_hash"],
                        cache["drug_x"].size(0),
                        cache["kinase_x"].size(0),
                        len(cache["train_records"]),
                        len(cache["valid_records"]),
                        cache["affinity_min"],
                        cache["affinity_max"],
                    )
                )

        base_model = MMCLKin(
            args.lstm_dropout,
            args.alpha,
            args.num_heads,
            hidden_dim=params["hidden_dim"],
            dropout_rate=params["dropout_rate"],
            n_head=8,
            smile_vocab=63,
            local_rank=device,
        )
        base_load_report = None
        if args.graph_node_feature_mode == "base_top10":
            base_model = base_model.to(device)
            base_load_report = load_base_only_from_checkpoint(
                base_model, args.base_checkpoint, device, min_loaded=20,
            )
            if base_load_report["missing_keys"] or base_load_report["shape_mismatch_keys"]:
                raise RuntimeError("base_top10 requires complete Base checkpoint loading")
            for parameter in base_model.parameters():
                parameter.requires_grad = False
            base_model.eval()
            feature_result = None
            if rank == 0:
                try:
                    entity_payload = load_or_build_frozen_entity_cache(
                        base_model, cache, args.base_checkpoint, args.data_path,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "jimei"),
                        base_collate, device,
                    )
                    feature_result = {"path": entity_payload["cache_path"], "error": None}
                except Exception as exc:
                    feature_result = {"path": None, "error": "{}: {}".format(type(exc).__name__, exc)}
            feature_result = broadcast_object(feature_result, src=0)
            if feature_result["error"] is not None:
                raise RuntimeError("frozen entity cache build failed: {}".format(feature_result["error"]))
            barrier()
            entity_payload = torch.load(feature_result["path"], map_location="cpu")
            for name in ("drug_base_x", "kinase_protein_base_x", "kinase_pocket_base_x"):
                cache[name] = entity_payload[name]
            cache["graph_node_feature_mode"] = "base_top10"
            cache["frozen_entity_signature"] = entity_payload["signature"]
            if rank == 0:
                print("frozen_entity_cache={} signature={} shapes={} audited_entities={} audit_scope={}".format(
                    feature_result["path"], entity_payload["signature"],
                    {name: tuple(cache[name].shape) for name in (
                        "drug_base_x", "kinase_protein_base_x", "kinase_pocket_base_x"
                    )}, len(entity_payload["audit_files"]), entity_payload["audit_scope"],
                ))
        if cache is not None:
            drug_input_dim = cache["drug_x"].size(1)
            kinase_input_dim = cache["kinase_x"].size(1)
            if args.graph_node_feature_mode == "base_top10":
                drug_input_dim += cache["drug_base_x"].size(1) + 20
                kinase_input_dim += cache["kinase_protein_base_x"].size(1) + cache["kinase_pocket_base_x"].size(1) + 20
                if rank == 0:
                    print("graph_node_input_dims drug={} kinase={} similarity_feature_topk=10".format(
                        drug_input_dim, kinase_input_dim,
                    ))
            model = MMCLKinWithHeteroView(
                base_model,
                drug_input_dim=drug_input_dim,
                kinase_input_dim=kinase_input_dim,
                graph_hidden_dim=args.graph_hidden_dim,
                graph_num_layers=args.graph_num_layers,
                pair_feature_dim=params["hidden_dim"] * 2,
                dropout=params["dropout_rate"],
                fusion_mode=args.fusion_mode,
                graph_max_delta=args.graph_max_delta,
                graph_relation_mode=args.graph_relation_mode,
                graph_affinity_mode=args.graph_affinity_mode,
                graph_affinity_message=args.graph_affinity_message,
                graph_affinity_source_beta=args.graph_affinity_source_beta,
            )
        else:
            if args.graph_only:
                raise ValueError("graph_only requires --use_hetero_graph")
            model = base_model
        if args.graph_only:
            if not args.base_checkpoint:
                raise ValueError("graph_only requires --base_checkpoint")
            if base_load_report is None:
                base_load_report = load_base_only_from_checkpoint(
                    raw_model(model).base_model,
                    args.base_checkpoint,
                    device,
                    min_loaded=20,
                )
            freeze_base_model(model)
        model = model.to(device)
        if args.distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

        if args.graph_only:
            graph_optimizer, graph_parameters = build_graph_optimizer(
                model, args.graph_lr, args.graph_weight_decay
            )
            optimizer = None
            scheduler = None
        else:
            optimizer = build_optimizer(model, args, params)
            scheduler = build_scheduler(optimizer, args, params)
            graph_optimizer = None
            graph_parameters = None
        scaler = GradScaler(enabled=args.use_amp)
        start_epoch = 0
        best_state = {
            "best_mse": 1e9,
            "best_pearson": -1e9,
            "best_epoch": -1,
            "best_fusion_mse": 1e9,
            "best_fusion_mse_epoch": -1,
            "best_fusion_pearson": -1e9,
            "best_fusion_pearson_epoch": -1,
            "best_base_mse": 1e9,
            "best_base_mse_epoch": -1,
            "best_base_pearson": -1e9,
            "best_base_pearson_epoch": -1,
            "early_stop_count": 0,
        }
        if args.resume:
            start_epoch, best_state = resume_checkpoint(
                args.resume, model, optimizer, scheduler, scaler, device, rank,
                expected_graph_config={
                    "graph_node_feature_mode": args.graph_node_feature_mode,
                    "graph_relation_mode": args.graph_relation_mode,
                    "graph_affinity_mode": args.graph_affinity_mode,
                    "graph_affinity_message": args.graph_affinity_message,
                    "graph_affinity_source_beta": args.graph_affinity_source_beta,
                    "graph_low_over_lambda": args.graph_low_over_lambda,
                    "frozen_entity_signature": cache.get("frozen_entity_signature") if cache is not None else None,
                },
                graph_optimizer=graph_optimizer,
            )
        monitor_kinases = []
        if cache is not None:
            if rank == 0:
                monitor_kinases = fixed_monitor_kinases(
                    cache["train_kinases"],
                    args.graph_monitor_kinase_count,
                    args.graph_monitor_seed,
                )
            monitor_kinases = list(broadcast_object(monitor_kinases, src=0))
        if rank == 0 and args.graph_only:
            print("base_only_load_report={}".format(base_load_report))
            print("graph_lr={:.8f}".format(args.graph_lr))
            print("graph_monitor_kinases={}".format(monitor_kinases))

        end_epoch = start_epoch + 1 if args.dry_run else args.EPOCHS
        for epoch in range(start_epoch, end_epoch):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.time()
            if rank == 0:
                print("\n======= epoch {} =======".format(epoch))
            if args.graph_only:
                base_stats = {"skipped": True, "reason": "graph_only"}
            else:
                base_stats = train_base_epoch(
                    model,
                    train_loader,
                    train_sampler,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    epoch,
                    args,
                    params,
                    rank,
                )
            graph_stats = None
            query_pairs: Set[Pair] = set()
            if cache is not None:
                if args.graph_query_mode == "pair":
                    if rank == 0:
                        query_pairs = select_query_pairs(
                            cache, args.graph_query_count, args.seed, epoch
                        )
                    query_pairs = set(broadcast_object(sorted(query_pairs), src=0))
                    support_graph, support_pairs = build_support_graph(cache, query_pairs)
                    if rank == 0:
                        print(
                            "graph query_hash={} query_pairs={} support_edges={}".format(
                                query_hash(query_pairs),
                                len(query_pairs),
                                support_graph[
                                    ("drug", "binds", "kinase")
                                ].edge_index.size(1),
                            )
                        )
                        for pair in sorted(query_pairs)[:5]:
                            print(
                                "query_check pair={} support_edge={} similar_edges_allowed=true "
                                "label_only_in_target=true".format(
                                    pair, pair in support_pairs
                                )
                            )
                    graph_stats = train_graph_epoch(
                        model,
                        cache,
                        query_pairs,
                        optimizer,
                        scaler,
                        device,
                        epoch,
                        args,
                        rank,
                        world_size,
                    )
                else:
                    if rank == 0:
                        kinase_rounds = select_pseudo_cold_kinase_rounds(
                            cache["train_kinases"],
                            args.graph_query_kinase_count,
                            args.graph_query_rounds,
                            args.seed,
                            epoch,
                            excluded_kinases=set(monitor_kinases),
                        )
                    else:
                        kinase_rounds = None
                    kinase_rounds = broadcast_object(kinase_rounds, src=0)
                    graph_stats = train_entity_graph_epoch(
                        model,
                        cache,
                        kinase_rounds,
                        graph_optimizer if args.graph_only else optimizer,
                        scaler,
                        device,
                        epoch,
                        args,
                        rank,
                        world_size,
                        graph_parameters,
                    )

            graph_monitor = {}
            if cache is not None and args.graph_monitor_kinase_count > 0:
                graph_monitor = evaluate_fixed_graph_monitor(
                    model,
                    cache,
                    monitor_kinases,
                    device,
                    epoch,
                    args,
                    rank,
                    world_size,
                )

            validation = validate(
                model,
                valid_loader,
                valid_sampler,
                cache,
                device,
                epoch,
                args,
                params,
                rank,
            )
            primary_metrics = (
                validation["fusion"] if cache is not None else validation["base"]
            )
            if scheduler is not None and args.scheduler == "ReduceLROnPlateau":
                scheduler.step(primary_metrics["mse"])

            if rank == 0:
                print("base_train={}".format(base_stats))
                if graph_stats is not None:
                    print("graph_train={}".format(graph_stats))
                if graph_monitor:
                    print("graph_monitor_metrics={}".format(graph_monitor))
                print("base_valid_metrics={}".format(validation["base"]))
                print("base_valid_diagnostics={}".format(validation["base_diagnostics"]))
                if cache is not None:
                    print("fusion_valid_metrics={}".format(validation["fusion"]))
                    print(
                        "fusion_valid_diagnostics={}".format(
                            validation["fusion_diagnostics"]
                        )
                    )

                base_metrics = validation["base"]
                fusion_metrics = validation["fusion"]
                base_mse_improved = base_metrics["mse"] < best_state["best_base_mse"]
                base_pearson_improved = (
                    base_metrics["pearson_value"] > best_state["best_base_pearson"]
                )
                fusion_mse_improved = fusion_metrics["mse"] < best_state["best_fusion_mse"]
                fusion_pearson_improved = (
                    fusion_metrics["pearson_value"] > best_state["best_fusion_pearson"]
                )
                primary_mse_improved = (
                    fusion_metrics["mse"]
                    < best_state["best_fusion_mse"]
                    - (args.graph_min_delta if args.graph_only else 0.0)
                )
                if base_mse_improved:
                    best_state["best_base_mse"] = float(base_metrics["mse"])
                    best_state["best_base_mse_epoch"] = epoch
                if base_pearson_improved:
                    best_state["best_base_pearson"] = float(base_metrics["pearson_value"])
                    best_state["best_base_pearson_epoch"] = epoch
                if fusion_mse_improved:
                    best_state["best_fusion_mse"] = float(fusion_metrics["mse"])
                    best_state["best_fusion_mse_epoch"] = epoch
                    best_state["best_mse"] = float(fusion_metrics["mse"])
                if fusion_pearson_improved:
                    best_state["best_fusion_pearson"] = float(
                        fusion_metrics["pearson_value"]
                    )
                    best_state["best_fusion_pearson_epoch"] = epoch
                    best_state["best_pearson"] = float(fusion_metrics["pearson_value"])
                    best_state["best_epoch"] = epoch
                if primary_mse_improved:
                    best_state["early_stop_count"] = 0
                else:
                    best_state["early_stop_count"] += 1

                checkpoint_common = dict(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_state=best_state,
                    args=args,
                    params=params,
                    cache=cache,
                    graph_optimizer=graph_optimizer,
                    monitor_state=graph_monitor,
                )
                if cache is None:
                    if base_mse_improved:
                        save_checkpoint(
                            os.path.join(save_dir, "{}_mse_best.pkl".format(args.model)),
                            **checkpoint_common,
                        )
                    if base_pearson_improved:
                        save_checkpoint(
                            os.path.join(save_dir, "{}_pearson_best.pkl".format(args.model)),
                            **checkpoint_common,
                        )
                else:
                    if base_mse_improved:
                        save_checkpoint(os.path.join(save_dir, "best_base_mse.pkl"), **checkpoint_common)
                    if base_pearson_improved:
                        save_checkpoint(os.path.join(save_dir, "best_base_pearson.pkl"), **checkpoint_common)
                    if fusion_mse_improved:
                        save_checkpoint(os.path.join(save_dir, "best_fusion_mse.pkl"), **checkpoint_common)
                    if fusion_pearson_improved:
                        save_checkpoint(os.path.join(save_dir, "best_fusion_pearson.pkl"), **checkpoint_common)
                save_checkpoint(os.path.join(save_dir, "last.pkl"), **checkpoint_common)
                patience = (
                    args.graph_early_stop_patience
                    if args.graph_only
                    else args.early_stop_patience
                )
                stop = 0 < patience <= best_state["early_stop_count"]
                elapsed = time.time() - epoch_start
                print(
                    "epoch_time={:.1f}s best_fusion_mse={:.6f} "
                    "best_fusion_mse_epoch={} best_fusion_pearson={:.6f} "
                    "best_fusion_pearson_epoch={} best_base_mse={:.6f} "
                    "best_base_pearson={:.6f} early_stop_count={}".format(
                        elapsed,
                        best_state["best_fusion_mse"],
                        best_state["best_fusion_mse_epoch"],
                        best_state["best_fusion_pearson"],
                        best_state["best_fusion_pearson_epoch"],
                        best_state["best_base_mse"],
                        best_state["best_base_pearson"],
                        best_state["early_stop_count"],
                    )
                )
            else:
                stop = False

            synced_state = broadcast_object(best_state if rank == 0 else None, src=0)
            best_state = synced_state
            stop = broadcast_bool(stop, device, src=0)
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                maximum = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(
                    "rank={} memory_gb allocated={:.3f} reserved={:.3f} max={:.3f}".format(
                        rank, allocated, reserved, maximum
                    )
                )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            barrier()
            if stop:
                if rank == 0:
                    print("early stop after epoch {}".format(epoch))
                break

        if rank == 0:
            print(
                "finished: best_fusion_mse={:.6f} best_fusion_pearson={:.6f} "
                "best_fusion_pearson_epoch={} "
                "output={}".format(
                    best_state["best_fusion_mse"],
                    best_state["best_fusion_pearson"],
                    best_state["best_fusion_pearson_epoch"],
                    save_dir,
                )
            )
        barrier()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if stdout_logger is not None:
            stdout_logger.close()
        if stderr_logger is not None:
            stderr_logger.close()
        cleanup_distributed()


PARAMS = {
    "batch_size": 128,
    "learning_rate": 0.0001,
    "lr_reduce_rate": 0.8,
    "dropout_rate": 0.3,
    "hidden_dim": 256,
    "beta1": 0.3,
    "beta2": 0.1,
    "beta3": 0.6,
    "beta4": 0.1,
}


if __name__ == "__main__":
    main(build_parser().parse_args(), PARAMS)
