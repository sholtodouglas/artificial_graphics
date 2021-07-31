# taken from https://github.com/facebookresearch/detr/blob/master/models/matcher.py

import jax
import flax.linen as nn
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from jax_resnet import pretrained_resnet, Sequential, slice_variables, ResNet50
from typing import Callable, Any, Optional
from flax import struct
from scipy.optimize import linear_sum_assignment



def center_to_corners_format(t):
    """
    Converts a tensor of bounding boxes of center format [(center_x, center_y, width, height)] to corners format
    [(x_0, y_0, x_1, y_1)].
    """

    x,y,w,h = t[:,0],t[:,1],t[:,2],t[:,3]
    x_0, y_0, x_1, y_1 =  x - 0.5*w, y - 0.5*h, x + 0.5*w, y + 0.5*h
    return jnp.stack([x_0, y_0, x_1,y_1]).T

def box_area(boxes: jnp.array) -> jnp.array:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.
    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = jnp.maximum(boxes1[:, jnp.newaxis, :2], boxes2[:, :2])  # [N,M,2]
    rb = jnp.minimum(boxes1[:, jnp.newaxis, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = jnp.clip(rb - lt, 0, None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.
    Returns:
        a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = jnp.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jnp.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = jnp.clip(rb - lt, 0, None)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class DetrHungarianMatcher(): # TODO maybe make as a module
    """
    This class computes an assignment between the targets and the predictions of the network.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1, fill_cost=0.2, rotation_cost=0.2):
        """
        Creates the matcher.
        Params:
            class_cost: This is the relative weight of the classification error in the matching cost
            bbox_cost: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            giou_cost: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.fill_cost = fill_cost
        self.rotation_cost = rotation_cost
        assert class_cost != 0 or bbox_cost != 0 or giou_cost != 0, "All costs of the Matcher can't be 0"

    def forward(self, outputs, targets):
        """
        Performs the matching.
        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                 objects in the target) containing the class labels "boxes": Tensor of dim [num_target_boxes, 4]
                 containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["logits"].shape[:2] # B, 100
        # We flatten to compute the cost matrices in a batch
        out_prob = nn.softmax(jnp.reshape(outputs["logits"], (bs*num_queries, -1)), -1)# [batch_size * num_queries, num_classes]
        out_bbox = jnp.reshape(outputs["pred_boxes"], (bs*num_queries, -1))  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        tgt_ids = jnp.concatenate([v["class_labels"] for v in targets]) # N_tgts
        tgt_bbox = jnp.concatenate([v["boxes"] for v in targets]) # N_tgts (B * per batch tgts)
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -out_prob[:, tgt_ids] # Out prob is B*Q, num_classes. This gets the -proba[target_class] for each of those heads
        # Compute the L1 cost between boxes
        n_outputs, n_targets = out_bbox.shape[0], tgt_bbox.shape[0]
        # bbox_cost = jnp.linalg.norm(jnp.tile(out_bbox, (n_targets, 1))- jnp.repeat(tgt_bbox, n_outputs, 0), 1, axis=-1) # L1 dist between this BBox and all the tgt bboxs B*NQ, N_tgts
        bbox_cost = jnp.linalg.norm(jnp.repeat(out_bbox, n_targets, 0)- jnp.tile(tgt_bbox, (n_outputs, 1)), 1, axis=-1) # L1 dist between this BBox and all the tgt bboxs B*NQ, N_tgts
        bbox_cost = bbox_cost.reshape(n_outputs, n_targets)
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(tgt_bbox))
        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = jnp.reshape(cost_matrix, (bs, num_queries, -1))
        sizes = [len(v["boxes"]) for v in targets]

        # To replicate torch split, we need the actual chunk indices not the lengths
        chunks = [0]
        for s in sizes[:-1]:
            chunks.append(s+chunks[-1])
            
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(chunks[1:], -1))] # this splits the cost matrix up, i.e if B 1 has 9 labels and 2 has 6, then solves the linear sum assignment
            
        # returns (i,j) where i is the head idx (out of 100) and j is the target idx out of N_tgts
        return [(jnp.array(i, dtype=jnp.int32), jnp.array(j, dtype=jnp.int32)) for i, j in indices]

def weighted_softmax_ce(logits, labels, weights):
    unweighted = labels * jax.nn.log_softmax(logits, axis=-1) # B,N_targets,N_classes+1
    return -jnp.sum(unweighted*weights) / jnp.sum(weights*labels) # replicates pytorch's weighted softmax. Divide by the sum of the true weights (as each example is multiplied by its weight)

def scatter_nd(original, indices, updates):
    key = tuple(jnp.moveaxis(indices, -1, 0))
    return original.at[key].set(updates)


class DetrLoss():
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        """
        Create the criterion.
        A note on the num_classes parameter (copied from original repo in detr.py): "the naming of the `num_classes`
        parameter of the criterion is somewhat misleading. it indeed corresponds to `max_obj_id + 1`, where max_obj_id
        is the maximum id for a class in your dataset. For example, COCO has a max_obj_id of 90, so we pass
        `num_classes` to be 91. As another example, for a dataset that has a single class with id 1, you should pass
        `num_classes` to be 2 (max_obj_id + 1). For more details on this, check the following discussion
        https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            num_classes: number of object categories, omitting the special no-object category.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.ce_weight = jnp.concatenate([jnp.ones(self.num_classes), jnp.array([self.eos_coef])]) # different coeff for null


    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        assert "logits" in outputs, "No logits were found in the outputs"
        src_logits = outputs["logits"] # B, 100, N_classes

        idx = self._get_src_permutation_idx(indices) #  batch indices ) (tensor([0, 0, 0, 0, ... 1, 1, 1]), tensor([ 2,  7, 11, ..., 27, 59]))
        target_classes_o = jnp.concatenate([t["class_labels"][J] for t, (_, J) in zip(targets, indices)]) # gets the class at each idx
        
        target_classes = jnp.ones(src_logits.shape[:2])* self.num_classes # default inits with the final 'no class' label (i.e, creates a B, N)

        
        target_classes = scatter_nd(target_classes, jnp.vstack(idx).astype(jnp.int32).T, target_classes_o)

        loss_ce = weighted_softmax_ce(src_logits, jax.nn.one_hot(target_classes, self.num_classes+1), self.ce_weight) # ([B, N_classes, N_heads]), [B, N_heads], empty_weight is just 1s for all classes but 0.1 for null class

        losses = {"loss_ce": loss_ce}

        return losses

    # TODO CAN WE SPECIFY NO GRAD
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]

        tgt_lengths = jnp.array([len(v["class_labels"]) for v in targets])
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = jnp.mean(jnp.abs(card_pred.astype(jnp.float32) - tgt_lengths.astype(jnp.float32)))
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.
        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs, "No predicted boxes found in outputs"
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        
        target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

        loss_bbox = jnp.abs(src_boxes - target_boxes)


        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        
        loss_giou = 1 - jnp.diag(
            generalized_box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses


    
        # Source (nn output)
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = jnp.concatenate([(jnp.ones(src.shape)*i).astype(jnp.int32) for i, (src, _) in enumerate(indices)]) # creates the batch indices (i.e 0,0,0,0,1,1) depending on how many tgts there are
        src_idx = jnp.concatenate([src.astype(jnp.int32) for (src, _) in indices]) # just gets the source indices that correspond
        
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = jnp.concatenate([(jnp.ones(tgt.shape)*i).astype(jnp.int32) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = jnp.concatenate([tgt.astype(jnp.int32) for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = { 
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"Loss {loss} not supported"
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"} # logits, pred_boxes

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher.forward(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = jnp.array([num_boxes], dtype=jnp.float32)
        
        num_boxes = jnp.clip(num_boxes, 1)
        print(num_boxes[0])

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Todo maybe re-include intermeidate auxiliary losses

        return losses



@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    output_dim: int
    share_embeddings: bool = False
    logits_via_embedding: bool = False
    dtype: Any = jnp.float32
    emb_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 100
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    deterministic: bool = False
    decode: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    posemb_init: Optional[Callable] = None
    seed: int = 42
        
        
@struct.dataclass
class TestConfig:
    """Test hyperparameters used to minimize obnoxious kwarg plumbing."""
    output_dim: int
    dtype: Any = jnp.float32
    emb_dim: int = 32
    num_heads: int = 8
    num_layers: int = 3
    qkv_dim: int = 32
    mlp_dim: int = 16
    max_len: int = 100
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    deterministic: bool = False
    decode: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    posemb_init: Optional[Callable] = None
    seed: int = 42



class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(cfg.mlp_dim,
                 dtype=cfg.dtype,
                 kernel_init=cfg.kernel_init,
                 bias_init=cfg.bias_init)(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    output = nn.Dense(actual_out_dim,
                      dtype=cfg.dtype,
                      kernel_init=cfg.kernel_init,
                      bias_init=cfg.bias_init)(x)
    output = nn.Dropout(rate=cfg.dropout_rate)(
        output, deterministic=cfg.deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               encoder_mask=None):
    """Applies Encoder1DBlock module.
    Args:
      inputs: input data.
      encoder_mask: encoder self-attention mask.
    Returns:
      output after transformer encoder block.
    """
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic)(x, encoder_mask)

    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = MlpBlock(config=cfg)(y)

    return x + y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.
    Args:
      targets: input data for decoder
      encoded: input data from encoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.
    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config


    x = nn.LayerNorm(dtype=cfg.dtype)(targets)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic,
        decode=cfg.decode)(x, decoder_mask)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = nn.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic)(
            y, encoded, encoder_decoder_mask)

    y = nn.Dropout(rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(dtype=cfg.dtype)(y)
    z = MlpBlock(config=cfg)(z)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Any = None

  @nn.compact
  def __call__(self,
               inputs,
               encoder_mask=None):
    """Applies Transformer model on the inputs.
    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      encoder_mask: decoder self-attention mask.
    Returns:
      output of a transformer encoder.
    """
    cfg = self.config

    x = inputs
    # Input Encoder
    for lyr in range(cfg.num_layers):
        x = Encoder1DBlock(config=cfg, name=f'encoderblock_{lyr}')(x, encoder_mask)

    encoded = nn.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Any = None

  @nn.compact
  def __call__(self,
               encoded,
               targets,
               decoder_mask=None,
               encoder_decoder_mask=None): # even if doing causal - you'll want all ones on the encoded part, and causal on the decoded part
    """Applies Transformer model on the inputs.
    Args:
      encoded: encoded input data from encoder.
      targets: target inputs.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.
    Returns:
      output of a transformer decoder.
    """
    cfg = self.config
    y = targets
    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      y = EncoderDecoder1DBlock(
          config=cfg, name=f'encoderdecoderblock_{lyr}')(y,encoded,decoder_mask=decoder_mask,encoder_decoder_mask=encoder_decoder_mask)
    y = nn.LayerNorm(dtype=cfg.dtype, name='encoderdecoder_norm')(y)
    return y


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation.
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  def setup(self):
    self.encoder = Encoder(config=self.config)
    self.decoder = Decoder(config=self.config)

  def __call__(self,
               inputs,
               targets,
               encoder_mask=None,
               decoder_mask=None):
    """Applies Transformer model on the inputs.
    Args:
      inputs: input data.
      targets: target data.
    Returns:
      logits array from full transformer.
    """
    encoded = self.encoder(inputs, encoder_mask=encoder_mask)
    
    if encoder_mask == None and decoder_mask == None:
        encoder_decoder_mask = None
    else:
        raise NotImplemented

    return self.decoder(encoded, targets, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask).astype(self.config.dtype)

class DETR_transformer(nn.Module):
    config: TransformerConfig
        
    def setup(self):
        self.backbone = Sequential(ResNet50(n_classes=1).layers[0:18])
        self.feature_conv = nn.Conv(features = self.config.mlp_dim, kernel_size = (1,1)) # 1D conv to convert resnet outputs down
        self.transformer = Transformer(self.config)
        self.linear_class = nn.Dense(self.config.output_dim+1, dtype=self.config.dtype, kernel_init=self.config.kernel_init,bias_init=self.config.bias_init) # +1 is for the empty class
        self.linear_bbox = nn.Dense(4, dtype=self.config.dtype, kernel_init=self.config.kernel_init,bias_init=self.config.bias_init)
        
        self.query_pos = self.param('queries', nn.initializers.uniform(scale=1), (100, self.config.mlp_dim), self.config.dtype)
        self.row_embed = self.param('row_embed', nn.initializers.uniform(scale=1), (50, self.config.mlp_dim//2), self.config.dtype)
        self.col_embed = self.param('col_embed', nn.initializers.uniform(scale=1), (50, self.config.mlp_dim//2), self.config.dtype)
        
    def __call__(self, image):
        x = self.backbone(image)

        x = self.feature_conv(x)
        B, H, W, EMB = x.shape[0], x.shape[1], x.shape[2], x.shape[3] # 0 is batch, 3 is feature

        col_embeds = jnp.repeat(self.col_embed[:W][jnp.newaxis,:,:],H, 0) #  H, W, embedding_size//2
        row_embeds = jnp.repeat(self.col_embed[:H][:,jnp.newaxis,:],W, 1) # H, W, embedding_size//2
        
        positional_embeds = jnp.concatenate([col_embeds,row_embeds], -1) # H, W, embedding_size

        positional_embeds_as_seq = jnp.reshape(positional_embeds, (1, H*W, EMB)) # H*W, embedding_size

        image_tiles_as_seq = jnp.reshape(x, (B, H*W, -1))
        
        queries = jnp.repeat(self.query_pos[jnp.newaxis, :, :], B, 0)
        
        
        x = self.transformer(positional_embeds_as_seq + 0.1*image_tiles_as_seq, queries)
        
        pred_logits = self.linear_class(x)
        pred_bbox = self.linear_bbox(x)
        
        

        return {'logits': pred_logits, 'pred_boxes':pred_bbox}
    
    
def load_pretrained_resnet(params):
    import torch 
    state_dict = torch.load('pretrained/resnet50.pth')
    ResNet50, variables = pretrained_resnet(50, state_dict)
    rn = ResNet50()
    idx = 18 # gives B, 7,7,2048
    backbone, backbone_variables = Sequential(rn.layers[0:idx]), slice_variables(variables, end=idx)
    mutable_dict = params.unfreeze()
    mutable_dict['batch_stats']['backbone'] = backbone_variables['batch_stats']
    mutable_dict['params']['backbone'] = backbone_variables['params']
    return mutable_dict