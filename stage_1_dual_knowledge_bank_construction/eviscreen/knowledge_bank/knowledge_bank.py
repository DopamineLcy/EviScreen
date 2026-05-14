"""Knowledge_Bank and Knowledge_Bank detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import eviscreen.knowledge_bank as knowledge_bank
from eviscreen.knowledge_bank.pos_embed import interpolate_pos_embed

LOGGER = logging.getLogger(__name__)


class Knowledge_Bank(torch.nn.Module):
    def __init__(self, device):
        """Knowledge_Bank anomaly detection class."""
        super(Knowledge_Bank, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_scorer_num_nn=1,
        featuresampler=knowledge_bank.sampler.IdentitySampler(),
        nn_method=knowledge_bank.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = knowledge_bank.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = knowledge_bank.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = knowledge_bank.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = knowledge_bank.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_scorer_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = knowledge_bank.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, provide_patch_metadata=False, norm=False):
        """Returns feature embeddings for images with optional metadata."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        original_shapes = [(f.shape[-2], f.shape[-1]) for f in features]

        '''
        features['layer2'].shape
        torch.Size([2, 512, 28, 28])

        features['layer3'].shape
        torch.Size([2, 1024, 14, 14])
        '''
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        '''
        features[0].shape
        torch.Size([2, 784, 512, 3, 3])

        features[1].shape
        torch.Size([2, 196, 1024, 3, 3])
        '''

        patch_metadata = []
        if provide_patch_metadata:
            batch_size = images.shape[0]
            for batch_idx in range(batch_size):
                patch_shape, orig_shape = patch_shapes[0], original_shapes[0]
                h_patches, w_patches = patch_shape
                for h_idx in range(h_patches):
                    for w_idx in range(w_patches):
                        h_start = h_idx * self.patch_maker.stride
                        w_start = w_idx * self.patch_maker.stride

                        metadata = {
                            'image_idx': batch_idx,
                            'patch_h': h_idx,
                            'patch_w': w_idx,
                            'orig_h_start': h_start,
                            'orig_w_start': w_start,
                            'patch_size': self.patch_maker.patchsize,
                            'feature_map_shape': orig_shape
                        }
                        patch_metadata.append(metadata)

        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]  # torch.Size([2, 14, 14, 1024, 3, 3])
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)  # torch.Size([2, 14, 14, 1024, 3, 3])
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)  # torch.Size([2, 28, 28, 1024, 3, 3])
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        '''
        features[0].shape
        torch.Size([1568, 512, 3, 3])
        
        features[1].shape
        torch.Size([1568, 1024, 3, 3])
        '''
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)  # torch.Size([1568 (28*28*B), 2, 1024])

        if norm:
            features[:, 0, :] = features[:, 0, :] / torch.norm(features[:, 0, :], p=2, dim=-1, keepdim=True)
            features[:, 1, :] = features[:, 1, :] / torch.norm(features[:, 1, :], p=2, dim=-1, keepdim=True)

        features = self.forward_modules["preadapt_aggregator"](features)  # torch.Size([1568 (28*28*B), 1024])

        result = _detach(features)
        if provide_patch_shapes and provide_patch_metadata:
            return result, patch_shapes, patch_metadata
        elif provide_patch_shapes:
            return result, patch_shapes
        elif provide_patch_metadata:
            return result, patch_metadata
        return result

    def fit(self, training_data):
        """Knowledge_Bank training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data, save_path=None):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, save_path)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, save_path=None):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        patch_scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        # global_features = []
        global_distances = []
        # global_query_nns = []
        # global_retrieved_features = []
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for idx, image in enumerate(data_iterator):
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    # masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _patch_scores, _masks, _features, _distances, _query_nns = self._predict(image)
                for score, patch_score, mask in zip(_scores, _patch_scores, _masks):
                    scores.append(score)
                    patch_scores.append(patch_score)
                    # masks.append(mask)
                
                # global_features.append(_features)
                global_distances.append(_distances)
                # global_query_nns.append(_query_nns)

                if save_path is not None:
                    np.save(f"{save_path}/features_{idx}.npy", _features)
                    np.save(f"{save_path}/distances_{idx}.npy", _distances)
                    np.save(f"{save_path}/query_nns_{idx}.npy", _query_nns)
        return scores, patch_scores, masks, labels_gt, masks_gt, global_distances

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores, distances, query_nns = self.anomaly_scorer.predict([features])
            image_scores = patch_scores
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [patch_score for patch_score in patch_scores], [mask for mask in masks], [feature for feature in features], [distance for distance in distances], [query_nn for query_nn in query_nns]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "knowledge_bank_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving Knowledge_Bank data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        knowledge_bank_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(knowledge_bank_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: knowledge_bank.common.FaissNN(False, 4),
        prepend: str = "",
        **kwargs,
    ) -> None:
        LOGGER.info("Loading and initializing Knowledge_Bank.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            knowledge_bank_params = pickle.load(load_file)
            if "anomaly_scorer_num_nn" in kwargs:
                knowledge_bank_params["anomaly_scorer_num_nn"] = kwargs["anomaly_scorer_num_nn"]

        knowledge_bank_params["backbone"] = knowledge_bank.backbones.load(
            knowledge_bank_params["backbone.name"]
        )
        if "backbone_path" in kwargs and kwargs["backbone_path"] is not None:
            checkpoint = torch.load(kwargs["backbone_path"], map_location="cpu")
            if "teacher" in checkpoint:
                checkpoint = checkpoint["teacher"]
            checkpoint = {
                key.replace("backbone.", "") if key.startswith("backbone.") else key: value
                for key, value in checkpoint.items()
            }

            interpolate_pos_embed(knowledge_bank_params["backbone"], checkpoint)
            msg = knowledge_bank_params["backbone"].load_state_dict(checkpoint, strict=False)
            LOGGER.info("Loaded backbone checkpoint with message: %s", msg)

        knowledge_bank_params["backbone"].name = knowledge_bank_params["backbone.name"]
        del knowledge_bank_params["backbone.name"]
        self.load(**knowledge_bank_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)

    def predict_with_traceability(self, data, traceability_manager=None):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_with_traceability(data, traceability_manager)
        return self._predict_with_traceability(data, traceability_manager)

    def _predict_dataloader_with_traceability(self, dataloader, traceability_manager=None):
        _ = self.forward_modules.eval()

        scores = []
        patch_scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        global_distances = []
        global_query_nns = []
        traced_sources = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for idx, image in enumerate(data_iterator):
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    image = image["image"]
                
                _scores, _patch_scores, _masks, _features, _distances, _query_nns = self._predict(image)
                
                for score, patch_score, mask in zip(_scores, _patch_scores, _masks):
                    scores.append(score)
                    patch_scores.append(patch_score)

                global_distances.append(_distances)
                global_query_nns.append(_query_nns)
                
                # ，
                if traceability_manager is not None:
                    # normal，abnormal
                    traced_batch = []
                    for query_nn in _query_nns:
                        if hasattr(traceability_manager, 'normal_traceability_map'):
                            traced_sources_batch = traceability_manager._trace_single_memory(
                                query_nn, traceability_manager.normal_traceability_map
                            )
                            traced_batch.append(traced_sources_batch)
                    traced_sources.append(traced_batch)

        return scores, patch_scores, masks, labels_gt, masks_gt, global_distances, global_query_nns, traced_sources


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
