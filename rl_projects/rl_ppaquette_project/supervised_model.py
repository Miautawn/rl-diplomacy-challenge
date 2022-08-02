import math
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_, xavier_uniform_, zeros_
import numpy as np

from utilities.utility_functions import pad_tensor
from settings import MODEL_DATA_PATHS, DATA_BLUEPRINT, DATA_FEATURES, H_PARAMETERS
from utilities.layers.graph_convolution import GraphConvolutionEncoder
from utilities.state_space_functions import ORDER_VOCABULARY_SIZE, PAD_ID


class Encoder(torch.nn.Module):
    def __init__(self, data_features: dict, h_params: dict):
        super(Encoder, self).__init__()

        self.h_params = h_params
        self.data_features = data_features

        # asserting safety checks
        assert self.h_params["n_graph_conv"] >= 2
        assert self.h_params["attn_size"] % 2 == 0

        # set the embedding matrix for powers and seasons
        # these will be used to condition FiLM
        self.power_embedding = nn.Embedding(
            self.data_features["n_powers"], self.h_params["power_emb_size"]
        )
        uniform_(self.power_embedding.weight, -1, 1)

        self.season_embedding = nn.Embedding(
            self.data_features["n_seasons"], self.h_params["season_emb_size"]
        )
        uniform_(self.season_embedding.weight, -1, 1)

        self.film_output_dims = [self.h_params["gcn_size"]] * (
            self.h_params["n_graph_conv"] - 1
        ) + [self.h_params["attn_size"] // 2]

        # creating FiLM layer to hold weights for board encoding
        self.board_encoding_film_weight_layer = nn.Linear(
            in_features=self.h_params["power_emb_size"]
            + self.h_params["season_emb_size"],
            out_features=2 * sum(self.film_output_dims),
            bias=True,
        )
        xavier_uniform_(self.board_encoding_film_weight_layer.weight)
        zeros_(self.board_encoding_film_weight_layer.bias)

        # creating FiLM layer to hold weights for previous orders
        self.prev_orders_encoding_film_weight_layer = nn.Linear(
            in_features=self.h_params["power_emb_size"]
            + self.h_params["season_emb_size"],
            out_features=2 * sum(self.film_output_dims),
            bias=True,
        )
        xavier_uniform_(self.prev_orders_encoding_film_weight_layer.weight)
        zeros_(self.prev_orders_encoding_film_weight_layer.bias)

        # creating graph convolution layer for encoding board state
        self.board_state_gcn_conv_layer = GraphConvolutionEncoder(
            n_features=self.data_features["n_locations"],
            input_size=self.data_features["n_location_features"],
            output_gcn_size=self.h_params["gcn_size"],
            final_output_gcn_size=self.h_params["attn_size"] // 2,
            n_graph_conv_layers=self.h_params["n_graph_conv"],
            batch_size=self.h_params["batch_size"],
        )

        # creating graph convolution layer for encoging previous orders
        self.prev_orders_gcn_conv_layer = GraphConvolutionEncoder(
            n_features=self.data_features["n_locations"],
            input_size=self.data_features["n_orders_features"],
            output_gcn_size=self.h_params["gcn_size"],
            final_output_gcn_size=self.h_params["attn_size"] // 2,
            n_graph_conv_layers=self.h_params["n_graph_conv"],
            batch_size=self.h_params["batch_size"],
        )

        # Creating order embedding vector (to embed order_ix)
        self.order_embedding = torch.empty(
            (ORDER_VOCABULARY_SIZE, self.h_params["order_emb_size"])
        ).uniform_(-1, 1)

        # Creating candidate embedding
        self.candidate_embedding = torch.empty(
            (ORDER_VOCABULARY_SIZE, self.h_params["lstm_size"] + 1)
        ).uniform_(-1, 1)

    def _postprocess_input(self, inputs: dict):
        """
        Post-processes the inputs by reshaping, padding and casting some features
        """

        decoder_inputs = inputs["decoder_inputs"]
        decoder_lengths = inputs["decoder_lengths"]
        candidates = inputs["candidates"]

        # recast some features to float dtype
        board_state = inputs["board_state"].to(torch.float32)
        board_alignments = inputs["board_alignments"].to(torch.float32)
        prev_orders_state = inputs["prev_orders_state"].to(torch.float32)

        # recast some features to long dtype
        current_power = inputs["current_power"].to(torch.int64)
        current_season = inputs["current_season"].to(torch.int64)

        # Reshaping board alignments to original form from flattened array
        board_alignments = torch.reshape(
            board_alignments,
            (self.h_params["batch_size"], -1, self.data_features["n_nodes"]),
        )

        board_alignments /= torch.maximum(
            torch.tensor(1.0), torch.sum(board_alignments, dim=-1, keepdims=True)
        )

        # Overriding dropout_rates if pholder('dropout_rate') > 0
        # TODO: figure out if this is even needed (the source code does not assign it as an output)
        # dropout_rates = torch.where(torch.gd(pholder('dropout_rate'), 0.),
        #                         true_fn=lambda: tf.zeros_like(dropout_rates) + pholder('dropout_rate'),
        #                         false_fn=lambda: dropout_rates)

        # Padding board_alignments, decoder_inputs and candidates (NOTE: this could be useless since we are doing batch padding)
        board_alignments = pad_tensor(
            board_alignments, axis=1, min_size=torch.max(decoder_lengths).item()
        )
        decoder_inputs = pad_tensor(decoder_inputs, axis=-1, min_size=2)
        candidates = pad_tensor(
            candidates, axis=-1, min_size=self.data_features["max_candidates"]
        )

        # Making sure all RNN lengths are at least 1
        # making a copy of original decoder lengths
        # TODO: remember, that this previously used detach(). I removed it thinking it was useless
        raw_decoder_lengths = decoder_lengths.clone()
        decoder_lengths = torch.maximum(torch.tensor(1), decoder_lengths)

        # Reshaping candidates
        candidates = torch.reshape(
            candidates,
            (self.h_params["batch_size"], -1, self.data_features["max_candidates"]),
        )
        candidates = candidates[
            :, : torch.max(decoder_lengths).item(), :
        ]  # torch.int - (b, n_locs, MAX_CANDIDATES)

        # Trimming to the maximum number of candidates
        candidate_lengths = torch.sum(
            (torch.gt(candidates, PAD_ID)), -1
        )  # int32 - (b,)
        max_candidate_length = max(1, torch.max(candidate_lengths).item())
        candidates = candidates[:, :, :max_candidate_length]

        # putting the modified inputs back into dict
        inputs["decoder_inputs"] = decoder_inputs
        inputs["decoder_lengths"] = decoder_lengths
        inputs["raw_decoder_lengths"] = raw_decoder_lengths
        inputs["max_candidate_length"] = max_candidate_length
        inputs["candidates"] = candidates
        inputs["board_state"] = board_state
        inputs["board_alignments"] = board_alignments
        inputs["prev_orders_state"] = prev_orders_state
        inputs["current_power"] = current_power
        inputs["current_season"] = current_season

        return inputs

    def forward(self, inputs: dict):

        # ======== Feature post-processing ========
        inputs = self._postprocess_input(inputs)

        current_power = inputs["current_power"]
        current_season = inputs["current_season"]
        board_state = inputs["board_state"]
        prev_orders_state = inputs["prev_orders_state"]

        # ======== Computing FiLM Gammas and Betas ========
        # TODO: move to a separate class/file

        current_power_mask = F.one_hot(current_power, self.data_features["n_powers"])
        current_power_embedding = torch.sum(
            self.power_embedding.weight[None] * current_power_mask[:, :, None], axis=1
        )

        current_season_mask = F.one_hot(current_season, self.data_features["n_seasons"])
        current_season_embedding = torch.sum(
            self.season_embedding.weight[None] * current_season_mask[:, :, None], axis=1
        )

        film_embedding_input = torch.cat(
            [current_power_embedding, current_season_embedding], dim=1
        )

        # FiLM for board state
        board_film_weights = self.board_encoding_film_weight_layer(
            film_embedding_input
        )[:, None, :]

        board_film_gammas, board_film_betas = torch.tensor_split(
            board_film_weights, 2, dim=2
        )
        board_film_gammas = torch.split(board_film_gammas, self.film_output_dims, dim=2)
        board_film_betas = torch.split(board_film_betas, self.film_output_dims, dim=2)

        # FiLM for previous orders
        prev_ord_film_weights = self.prev_orders_encoding_film_weight_layer(
            film_embedding_input
        )[:, None, :]
        prev_ord_film_weights = torch.tile(
            prev_ord_film_weights, (self.data_features["n_prev_orders"], 1, 1)
        )

        prev_ord_film_gammas, prev_ord_film_betas = torch.tensor_split(
            prev_ord_film_weights, 2, dim=2
        )
        prev_ord_film_gammas = torch.split(
            prev_ord_film_gammas, self.film_output_dims, dim=2
        )
        prev_ord_film_betas = torch.split(
            prev_ord_film_betas, self.film_output_dims, dim=2
        )

        # ======== Encoding board & prev_order states via graph convolution ========

        # Encoding board_state
        board_state_0yr_conv = self.board_state_gcn_conv_layer(
            board_state, film_gammas=board_film_gammas, film_betas=board_film_betas
        )

        # Encoding prev_orders
        prev_orders_state = torch.reshape(
            prev_orders_state,
            (
                self.h_params["batch_size"] * self.data_features["n_prev_orders"],
                self.data_features["n_nodes"],
                self.data_features["n_orders_features"],
            ),
        )

        prev_ord_conv = self.prev_orders_gcn_conv_layer(
            prev_orders_state,
            film_gammas=prev_ord_film_gammas,
            film_betas=prev_ord_film_betas,
        )

        # Splitting into (b, n_prev_orders, N_NODES, attn_size // 2)
        # and reducing over mean of n_prev_orders
        prev_ord_conv = torch.reshape(
            prev_ord_conv,
            [
                self.h_params["batch_size"],
                self.data_features["n_prev_orders"],
                self.data_features["n_nodes"],
                self.h_params["attn_size"] // 2,
            ],
        )

        prev_ord_conv = torch.mean(prev_ord_conv, dim=1)

        # Concatenating the current board conv with the prev ord conv
        # The final board_state_conv should be of dimension (b, NB_NODE, attn_size)
        board_state_conv = torch.cat((board_state_0yr_conv, prev_ord_conv), dim=-1)

        return {
            "board_alignments": inputs["board_alignments"],
            "decoder_inputs": inputs["decoder_inputs"],
            "raw_decoder_lengths": inputs["raw_decoder_lengths"],
            "decoder_lengths": inputs["decoder_lengths"],
            "candidates": inputs["candidates"],
            "max_candidate_length": inputs["max_candidate_length"],
            "board_state_conv": board_state_conv,
            "board_state_0yr_conv": board_state_0yr_conv,
            "prev_ord_conv": prev_ord_conv,
            "order_embedding": self.order_embedding,
            "candidate_embedding": self.candidate_embedding,
        }
