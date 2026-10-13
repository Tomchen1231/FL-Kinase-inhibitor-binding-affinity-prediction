import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, HeteroConv


GRAPH_CACHE_VERSION = 1
DRUG_RELATION = ("drug", "similar", "drug")
KINASE_RELATION = ("kinase", "similar", "kinase")
BINDS_RELATION = ("drug", "binds", "kinase")
REVERSE_BINDS_RELATION = ("kinase", "rev_binds", "drug")
RELATIONS = (
    DRUG_RELATION,
    KINASE_RELATION,
    BINDS_RELATION,
    REVERSE_BINDS_RELATION,
)
Pair = Tuple[int, int]
Record = Tuple[int, int, float, str]


def parse_sample_ids(path: str) -> Pair:
    parts = os.path.basename(path).split("_")
    if len(parts) < 5:
        raise ValueError("unexpected 3DKDavis filename: {}".format(path))
    try:
        return int(parts[2]), int(parts[4])
    except ValueError as exc:
        raise ValueError("cannot parse drug/protein IDs from {}".format(path)) from exc


def _pool_token_feature(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.dim() == 3:
        if tokens.size(0) != 1:
            raise ValueError("expected a singleton leading token batch dimension")
        tokens = tokens.squeeze(0)
    if tokens.dim() != 2:
        raise ValueError("expected [tokens, features], got {}".format(tuple(tokens.shape)))
    pooled_tokens = tokens[1:-1] if tokens.size(0) > 2 else tokens
    return pooled_tokens.mean(dim=0).detach().cpu().float().contiguous()


def _file_list_signature(
    data_path: str,
    train_files: Sequence[str],
    valid_files: Sequence[str],
    csv_path: str,
    graph_params: Dict,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(GRAPH_CACHE_VERSION).encode("utf-8"))
    digest.update(os.path.abspath(data_path).encode("utf-8"))
    for name in sorted(set(train_files) | set(valid_files)):
        full_path = os.path.join(data_path, name)
        stat = os.stat(full_path)
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    csv_stat = os.stat(csv_path)
    digest.update(os.path.abspath(csv_path).encode("utf-8"))
    digest.update(str(csv_stat.st_size).encode("ascii"))
    digest.update(str(csv_stat.st_mtime_ns).encode("ascii"))
    digest.update(json.dumps(graph_params, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _select_neighbors(
    similarities: torch.Tensor,
    candidate_indices: Sequence[int],
    topk: int,
    threshold: Optional[float],
    excluded_index: Optional[int] = None,
) -> List[Tuple[int, float]]:
    candidates = [idx for idx in candidate_indices if idx != excluded_index]
    if not candidates:
        return []
    scores = similarities[candidates]
    count = min(max(1, topk), len(candidates))
    values, positions = torch.topk(scores, k=count)
    selected = [
        (candidates[int(position)], float(value.clamp(0.0, 1.0).item()))
        for value, position in zip(values, positions)
        if threshold is None or float(value.item()) >= threshold
    ]
    if not selected:
        best_position = int(torch.argmax(scores).item())
        best_index = candidates[best_position]
        selected = [(best_index, float(scores[best_position].clamp(0.0, 1.0).item()))]
    return selected


def build_similarity_edges(
    features: torch.Tensor,
    train_indices: Sequence[int],
    valid_indices: Sequence[int],
    topk: int,
    threshold: Optional[float],
    strict_inductive: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normalized = F.normalize(features.float(), dim=1)
    similarity = torch.mm(normalized, normalized.t())
    train_set = set(train_indices)
    valid_only = sorted(set(valid_indices) - train_set)
    edges: Dict[Tuple[int, int], float] = {}

    if strict_inductive:
        for source in sorted(train_set):
            for target, weight in _select_neighbors(
                similarity[source], sorted(train_set), topk, threshold, source
            ):
                edges[(source, target)] = max(weight, edges.get((source, target), 0.0))
                edges[(target, source)] = max(weight, edges.get((target, source), 0.0))
        for target in valid_only:
            for source, weight in _select_neighbors(
                similarity[target], sorted(train_set), topk, threshold
            ):
                edges[(source, target)] = max(weight, edges.get((source, target), 0.0))
    else:
        all_indices = list(range(features.size(0)))
        for source in all_indices:
            for target, weight in _select_neighbors(
                similarity[source], all_indices, topk, threshold, source
            ):
                edges[(source, target)] = max(weight, edges.get((source, target), 0.0))
                edges[(target, source)] = max(weight, edges.get((target, source), 0.0))

    ordered = sorted(edges.items())
    if not ordered:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    edge_index = torch.tensor([edge for edge, _ in ordered], dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([weight for _, weight in ordered], dtype=torch.float32)
    return edge_index, edge_weight


class EntityFeatureCacheBuilder:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def build(
        self, files: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], Dict[int, int]]:
        representatives_drug: Dict[int, str] = {}
        representatives_kinase: Dict[int, str] = {}
        for filename in sorted(files):
            drug_id, kinase_id = parse_sample_ids(filename)
            representatives_drug.setdefault(drug_id, filename)
            representatives_kinase.setdefault(kinase_id, filename)

        drug_ids = sorted(representatives_drug)
        kinase_ids = sorted(representatives_kinase)
        drug_id_to_index = {raw_id: index for index, raw_id in enumerate(drug_ids)}
        kinase_id_to_index = {raw_id: index for index, raw_id in enumerate(kinase_ids)}
        drug_features: List[torch.Tensor] = []
        kinase_features: List[torch.Tensor] = []

        for raw_id in drug_ids:
            sample = torch.load(
                os.path.join(self.data_path, representatives_drug[raw_id]),
                map_location="cpu",
            )
            drug_features.append(_pool_token_feature(sample.mol_embedding))
        for raw_id in kinase_ids:
            sample = torch.load(
                os.path.join(self.data_path, representatives_kinase[raw_id]),
                map_location="cpu",
            )
            kinase_features.append(_pool_token_feature(sample.pro_token_repre))

        drug_x = torch.stack(drug_features).float().contiguous()
        kinase_x = torch.stack(kinase_features).float().contiguous()
        return drug_x, kinase_x, drug_id_to_index, kinase_id_to_index


class HeteroGraphBuilder:
    def __init__(
        self,
        data_path: str,
        csv_path: str,
        cache_path: str,
        drug_topk: int,
        kinase_topk: int,
        drug_sim_threshold: Optional[float],
        kinase_sim_threshold: Optional[float],
        strict_inductive_graph: bool,
    ) -> None:
        self.data_path = data_path
        self.csv_path = csv_path
        self.cache_path = cache_path
        self.graph_params = {
            "drug_topk": int(drug_topk),
            "kinase_topk": int(kinase_topk),
            "drug_sim_threshold": drug_sim_threshold,
            "kinase_sim_threshold": kinase_sim_threshold,
            "strict_inductive_graph": bool(strict_inductive_graph),
        }

    def load_or_build(
        self,
        train_files: Sequence[str],
        valid_files: Sequence[str],
        rebuild: bool = False,
    ) -> Dict:
        signature = _file_list_signature(
            self.data_path,
            train_files,
            valid_files,
            self.csv_path,
            self.graph_params,
        )
        if not rebuild and os.path.exists(self.cache_path):
            cached = torch.load(self.cache_path, map_location="cpu")
            if (
                cached.get("version") == GRAPH_CACHE_VERSION
                and cached.get("data_signature") == signature
            ):
                return cached

        all_files = sorted(set(train_files) | set(valid_files))
        feature_builder = EntityFeatureCacheBuilder(self.data_path)
        drug_x, kinase_x, drug_map, kinase_map = feature_builder.build(all_files)
        dataframe = pd.read_csv(self.csv_path)
        required_columns = {"drug_id", "protein_id", "affinity"}
        missing = required_columns - set(dataframe.columns)
        if missing:
            raise KeyError("CSV missing required columns: {}".format(sorted(missing)))
        affinity_lookup = {
            (int(row.drug_id), int(row.protein_id)): float(row.affinity)
            for row in dataframe.itertuples()
        }

        train_records = self._records(train_files, affinity_lookup)
        valid_records = self._records(valid_files, affinity_lookup)
        train_drugs = sorted({record[0] for record in train_records})
        valid_drugs = sorted({record[0] for record in valid_records})
        train_kinases = sorted({record[1] for record in train_records})
        valid_kinases = sorted({record[1] for record in valid_records})
        drug_train_indices = [drug_map[item] for item in train_drugs]
        drug_valid_indices = [drug_map[item] for item in valid_drugs]
        kinase_train_indices = [kinase_map[item] for item in train_kinases]
        kinase_valid_indices = [kinase_map[item] for item in valid_kinases]

        drug_edge_index, drug_edge_weight = build_similarity_edges(
            drug_x,
            drug_train_indices,
            drug_valid_indices,
            self.graph_params["drug_topk"],
            self.graph_params["drug_sim_threshold"],
            self.graph_params["strict_inductive_graph"],
        )
        kinase_edge_index, kinase_edge_weight = build_similarity_edges(
            kinase_x,
            kinase_train_indices,
            kinase_valid_indices,
            self.graph_params["kinase_topk"],
            self.graph_params["kinase_sim_threshold"],
            self.graph_params["strict_inductive_graph"],
        )
        train_affinities = torch.tensor(
            [record[2] for record in train_records], dtype=torch.float32
        )
        affinity_min = float(train_affinities.min().item())
        affinity_max = float(train_affinities.max().item())
        cache_hash = hashlib.sha256(
            (signature + json.dumps(self.graph_params, sort_keys=True)).encode("utf-8")
        ).hexdigest()
        cache = {
            "version": GRAPH_CACHE_VERSION,
            "data_signature": signature,
            "cache_hash": cache_hash,
            "graph_params": self.graph_params,
            "drug_x": drug_x,
            "kinase_x": kinase_x,
            "drug_id_to_index": drug_map,
            "kinase_id_to_index": kinase_map,
            "train_drugs": train_drugs,
            "valid_drugs": valid_drugs,
            "train_kinases": train_kinases,
            "valid_kinases": valid_kinases,
            "train_records": train_records,
            "valid_records": valid_records,
            "drug_edge_index": drug_edge_index,
            "drug_edge_weight": drug_edge_weight,
            "kinase_edge_index": kinase_edge_index,
            "kinase_edge_weight": kinase_edge_weight,
            "affinity_min": affinity_min,
            "affinity_max": affinity_max,
            "affinity_normalization_source": "train_only",
        }
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save(cache, self.cache_path)
        return cache

    def _records(
        self, files: Sequence[str], affinity_lookup: Dict[Pair, float]
    ) -> List[Record]:
        records: List[Record] = []
        for filename in sorted(files):
            pair = parse_sample_ids(filename)
            if pair not in affinity_lookup:
                raise KeyError("missing affinity for pair {}".format(pair))
            records.append((pair[0], pair[1], affinity_lookup[pair], filename))
        return records


def validate_no_leakage(cache: Dict, label: str) -> None:
    train_pairs = {(record[0], record[1]) for record in cache["train_records"]}
    valid_pairs = {(record[0], record[1]) for record in cache["valid_records"]}
    if not train_pairs.isdisjoint(valid_pairs):
        raise AssertionError("train and valid pairs overlap")
    if label == "new_drug" and not set(cache["train_drugs"]).isdisjoint(
        cache["valid_drugs"]
    ):
        raise AssertionError("new_drug split leaks drug entities")
    if label == "new_kinase" and not set(cache["train_kinases"]).isdisjoint(
        cache["valid_kinases"]
    ):
        raise AssertionError("new_kinase split leaks kinase entities")
    if label == "both_new":
        if not set(cache["train_drugs"]).isdisjoint(cache["valid_drugs"]):
            raise AssertionError("both_new split leaks drug entities")
        if not set(cache["train_kinases"]).isdisjoint(cache["valid_kinases"]):
            raise AssertionError("both_new split leaks kinase entities")
    if cache.get("affinity_normalization_source") != "train_only":
        raise AssertionError("affinity normalization must use training labels only")


def build_support_graph(
    cache: Dict,
    excluded_pairs: Optional[Set[Pair]] = None,
    excluded_kinases: Optional[Set[int]] = None,
    simulate_new_kinase: bool = False,
    return_stats: bool = False,
):
    excluded_pairs = excluded_pairs or set()
    excluded_kinases = excluded_kinases or set()
    valid_pairs = {(record[0], record[1]) for record in cache["valid_records"]}
    support_records = [
        record
        for record in cache["train_records"]
        if (record[0], record[1]) not in excluded_pairs
        and (not simulate_new_kinase or record[1] not in excluded_kinases)
    ]
    support_pairs = {(record[0], record[1]) for record in support_records}
    if not support_pairs.isdisjoint(excluded_pairs):
        raise AssertionError("query pairs remain in the support graph")
    if not support_pairs.isdisjoint(valid_pairs):
        raise AssertionError("validation affinity edges entered the support graph")
    if simulate_new_kinase and any(record[1] in excluded_kinases for record in support_records):
        raise AssertionError("pseudo-cold kinase affinity edges remain in support graph")

    graph = HeteroData()
    graph["drug"].x = cache["drug_x"]
    graph["kinase"].x = cache["kinase_x"]
    graph[DRUG_RELATION].edge_index = cache["drug_edge_index"]
    graph[DRUG_RELATION].edge_weight = cache["drug_edge_weight"]

    drug_map = cache["drug_id_to_index"]
    kinase_map = cache["kinase_id_to_index"]
    excluded_kinase_indices = {
        kinase_map[item] for item in excluded_kinases if item in kinase_map
    }
    kinase_edge_index = cache["kinase_edge_index"]
    kinase_edge_weight = cache["kinase_edge_weight"]
    original_incoming = 0
    if simulate_new_kinase and excluded_kinase_indices and kinase_edge_index.numel() > 0:
        source = kinase_edge_index[0]
        target = kinase_edge_index[1]
        pseudo = torch.tensor(
            sorted(excluded_kinase_indices), dtype=torch.long, device=source.device
        )
        source_is_pseudo = (source.unsqueeze(1) == pseudo.unsqueeze(0)).any(dim=1)
        target_is_pseudo = (target.unsqueeze(1) == pseudo.unsqueeze(0)).any(dim=1)
        original_incoming = int((~source_is_pseudo & target_is_pseudo).sum().item())
        keep = ~source_is_pseudo
        kinase_edge_index = kinase_edge_index[:, keep]
        kinase_edge_weight = kinase_edge_weight[keep]
        if bool((kinase_edge_index[0].unsqueeze(1) == pseudo.unsqueeze(0)).any()):
            raise AssertionError("pseudo-cold outgoing similarity edges remain")
    graph[KINASE_RELATION].edge_index = kinase_edge_index
    graph[KINASE_RELATION].edge_weight = kinase_edge_weight

    if support_records:
        source = [drug_map[record[0]] for record in support_records]
        target = [kinase_map[record[1]] for record in support_records]
        edge_index = torch.tensor([source, target], dtype=torch.long)
        raw_affinities = torch.tensor(
            [record[2] for record in support_records], dtype=torch.float32
        )
        support_affinity_min = float(raw_affinities.min().item())
        support_affinity_max = float(raw_affinities.max().item())
        denominator = support_affinity_max - support_affinity_min
        if denominator == 0:
            edge_weight = torch.ones(len(support_records), dtype=torch.float32)
        else:
            edge_weight = torch.tensor(
                [
                    (record[2] - support_affinity_min) / denominator
                    for record in support_records
                ],
                dtype=torch.float32,
            ).clamp_(0.0, 1.0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        support_affinity_min = None
        support_affinity_max = None
    graph[BINDS_RELATION].edge_index = edge_index
    graph[BINDS_RELATION].edge_weight = edge_weight
    graph[REVERSE_BINDS_RELATION].edge_index = edge_index.flip(0)
    graph[REVERSE_BINDS_RELATION].edge_weight = edge_weight.clone()

    stats = {
        "support_affinity_min": support_affinity_min,
        "support_affinity_max": support_affinity_max,
        "support_edge_count": int(edge_index.size(1)),
        "pseudo_cold_incoming_similarity_edges": int(original_incoming),
        "pseudo_cold_outgoing_similarity_edges": 0,
        "pseudo_cold_no_incoming_kinases": [],
    }
    if simulate_new_kinase and excluded_kinase_indices:
        kin_edges = graph[KINASE_RELATION].edge_index
        for raw_kinase in sorted(excluded_kinases):
            if raw_kinase not in kinase_map:
                continue
            index = kinase_map[raw_kinase]
            incoming = int(((kin_edges[0] != index) & (kin_edges[1] == index)).sum().item())
            if incoming == 0:
                stats["pseudo_cold_no_incoming_kinases"].append(raw_kinase)
    if return_stats:
        return graph, support_pairs, stats
    return graph, support_pairs


def records_for_pairs(records: Sequence[Record], pairs: Set[Pair]) -> List[Record]:
    return [record for record in records if (record[0], record[1]) in pairs]


class WeightedHeteroGraphEncoder(nn.Module):
    def __init__(
        self,
        drug_input_dim: int,
        kinase_input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.ModuleDict(
            {
                "drug": nn.Linear(drug_input_dim, hidden_dim),
                "kinase": nn.Linear(kinase_input_dim, hidden_dim),
            }
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        relation: GraphConv(
                            (hidden_dim, hidden_dim), hidden_dim, aggr="add"
                        )
                        for relation in RELATIONS
                    },
                    aggr="sum",
                )
            )
            self.norms.append(
                nn.ModuleDict(
                    {
                        "drug": nn.LayerNorm(hidden_dim),
                        "kinase": nn.LayerNorm(hidden_dim),
                    }
                )
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: HeteroData) -> Dict[str, torch.Tensor]:
        hidden = {
            node_type: self.input_projection[node_type](graph[node_type].x)
            for node_type in ("drug", "kinase")
        }
        for conv, norms in zip(self.convs, self.norms):
            updated = conv(
                hidden,
                graph.edge_index_dict,
                edge_weight_dict=graph.edge_weight_dict,
            )
            hidden = {
                node_type: norms[node_type](
                    hidden[node_type] + self.dropout(F.elu(updated[node_type]))
                )
                for node_type in ("drug", "kinase")
            }
        return hidden


class GraphPairFusionHead(nn.Module):
    def __init__(
        self,
        graph_hidden_dim: int,
        pair_feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.graph_projection = nn.Sequential(
            nn.Linear(graph_hidden_dim * 4, pair_feature_dim),
            nn.LayerNorm(pair_feature_dim),
            nn.ELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(pair_feature_dim * 2),
            nn.Linear(pair_feature_dim * 2, pair_feature_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_feature_dim, pair_feature_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_feature_dim // 2, 1),
        )

    def forward(
        self,
        pair_feature: torch.Tensor,
        drug_feature: torch.Tensor,
        kinase_feature: torch.Tensor,
    ) -> torch.Tensor:
        graph_pair = torch.cat(
            (
                drug_feature,
                kinase_feature,
                drug_feature * kinase_feature,
                torch.abs(drug_feature - kinase_feature),
            ),
            dim=1,
        )
        projected_graph_pair = self.graph_projection(graph_pair)
        return self.fusion(torch.cat((pair_feature, projected_graph_pair), dim=1))


class GraphResidualFusionHead(GraphPairFusionHead):
    def __init__(
        self,
        graph_hidden_dim: int,
        pair_feature_dim: int,
        dropout: float,
        graph_max_delta: float,
    ) -> None:
        super().__init__(graph_hidden_dim, pair_feature_dim, dropout)
        self.graph_max_delta = float(graph_max_delta)
        final_layer = self.fusion[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        pair_feature: torch.Tensor,
        drug_feature: torch.Tensor,
        kinase_feature: torch.Tensor,
        base_prediction: torch.Tensor,
        return_delta: bool = False,
    ):
        raw_delta = super().forward(pair_feature, drug_feature, kinase_feature)
        graph_delta = self.graph_max_delta * torch.tanh(raw_delta)
        fusion_prediction = base_prediction + graph_delta
        if return_delta:
            return fusion_prediction, graph_delta
        return fusion_prediction


class MMCLKinWithHeteroView(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        drug_input_dim: int,
        kinase_input_dim: int,
        graph_hidden_dim: int,
        graph_num_layers: int,
        pair_feature_dim: int,
        dropout: float,
        fusion_mode: str = "direct_concat",
        graph_max_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.fusion_mode = fusion_mode
        self.hetero_encoder = WeightedHeteroGraphEncoder(
            drug_input_dim,
            kinase_input_dim,
            graph_hidden_dim,
            graph_num_layers,
            dropout,
        )
        if fusion_mode == "direct_concat":
            self.fusion_head = GraphPairFusionHead(
                graph_hidden_dim, pair_feature_dim, dropout
            )
        elif fusion_mode == "residual":
            self.fusion_head = GraphResidualFusionHead(
                graph_hidden_dim, pair_feature_dim, dropout, graph_max_delta
            )
        else:
            raise ValueError("unsupported fusion_mode: {}".format(fusion_mode))

    def forward(
        self,
        x_feats,
        graph_data: Optional[HeteroData] = None,
        use_graph: bool = False,
        graph_update_base: bool = True,
    ):
        if not use_graph:
            return self.base_model(x_feats)
        if graph_data is None:
            raise ValueError("graph_data is required when use_graph=True")
        if graph_update_base:
            base_outputs = self.base_model(x_feats, return_pair_feature=True)
        else:
            with torch.no_grad():
                base_outputs = self.base_model(x_feats, return_pair_feature=True)
        base_prediction = base_outputs[0]
        pair_feature = base_outputs[-1]
        if not graph_update_base:
            base_prediction = base_prediction.detach()
            pair_feature = pair_feature.detach()
        node_features = self.hetero_encoder(graph_data)
        drug_feature = node_features["drug"][x_feats.drug_node_index]
        kinase_feature = node_features["kinase"][x_feats.kinase_node_index]
        if self.fusion_mode == "residual":
            fusion_prediction = self.fusion_head(
                pair_feature, drug_feature, kinase_feature, base_prediction
            )
        else:
            fusion_prediction = self.fusion_head(
                pair_feature, drug_feature, kinase_feature
            )
        return (
            base_prediction,
            fusion_prediction,
            *base_outputs[1:-1],
            base_outputs[-1],
        )
