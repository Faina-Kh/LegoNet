"""One-batch forward and loss composition for LegoNet training."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

import config


@dataclass
class LossResult:
    """Total differentiable loss and scalar component values for one batch."""

    total: Any
    values: Dict[str, float] = field(default_factory=dict)


def _mean_value(loss: Any) -> Any:
    """Reduce a loss tensor using the runner's existing mean behavior."""
    return loss.mean()


def _detection_losses(raw_losses: Dict[str, Any], values: Dict[str, float]) -> Any:
    """Reduce detection losses and add their scalar values."""
    classification = raw_losses.get("classification")
    if classification is None:
        values.update(classification=-1, regression=-1, detection=-1)
        return None

    classification = _mean_value(classification)
    regression = _mean_value(raw_losses["regression"])
    values["classification"] = classification.item()
    values["regression"] = regression.item()
    values["detection"] = values["classification"] + values["regression"]
    return classification + regression


def combine_losses(raw_losses: Dict[str, Any], args: Any) -> LossResult:
    """Compose the current weighted total loss for a configured network branch."""
    values = {}
    network_type = args.network_type

    if network_type == "bbox_detection":
        total = _detection_losses(raw_losses, values)
        return LossResult(total, values)

    if network_type == "counting_lean":
        l1_loss = raw_losses.get("l1_estimation")
        maps_loss = raw_losses.get("maps")
        if l1_loss is None or maps_loss is None:
            return LossResult(None, {"l1_estimation": -1, "maps": -1})
        l1_loss = _mean_value(args.loss_weight * l1_loss)
        maps_loss = _mean_value(maps_loss)
        values.update(l1_estimation=l1_loss.item(), maps=maps_loss.item())
        return LossResult(l1_loss + maps_loss, values)

    if network_type == "counting_reg":
        regression = _mean_value(args.loss_weight * raw_losses["reg_estimation"])
        values["reg_estimation"] = regression.item()
        return LossResult(regression, values)

    detection = _detection_losses(raw_losses, values)
    estimate_type = config.AttributeEstimation.estimate_type

    if network_type == "both":
        if estimate_type == "withKeyPoints":
            counting = raw_losses.get("l1_counting")
            maps = raw_losses.get("maps")
            if counting is None or maps is None:
                return LossResult(detection, values)
            counting = _mean_value(counting)
            maps = _mean_value(maps)
            values.update(l1_counting=counting.item(), maps=maps.item())
            attribute_total = counting + maps
        else:
            counting = raw_losses.get("counting")
            if counting is None:
                return LossResult(detection, values)
            counting = _mean_value(counting)
            values["counting"] = counting.item()
            attribute_total = counting
        total = attribute_total if detection is None else detection + attribute_total
        return LossResult(total, values)

    color = raw_losses.get("color")
    length = raw_losses.get("length")
    diameter = raw_losses.get("diameter")
    maps = raw_losses.get("maps")
    required = [color, length, diameter]
    if estimate_type == "withKeyPoints":
        required.append(maps)
    if any(loss is None for loss in required):
        values.update(color=-1, length=-1, diameter=-1)
        if estimate_type == "withKeyPoints":
            values["maps"] = -1
        return LossResult(detection, values)

    color = args.color_loss_weight * color
    diameter = args.dia_loss_weight * diameter
    attribute_total = color + length + diameter
    if estimate_type == "withKeyPoints":
        maps = args.maps_loss_weight * maps
        attribute_total = attribute_total + maps
        values["maps"] = maps.item()
    attribute_total = _mean_value(attribute_total)
    values.update(
        color=color.item(),
        length=length.item(),
        diameter=diameter.item(),
    )
    total = attribute_total if detection is None else detection + attribute_total
    return LossResult(total, values)


def forward_losses(
    model: Any,
    data: Dict[str, Any],
    args: Any,
    sampler_group: Any,
) -> Dict[str, Any]:
    """Run the model and name its branch-specific raw loss outputs."""
    image = data["img"].to(config.General.device).float()
    if args.network_type == "bbox_detection":
        classification, regression = model(
            [image, data["bbox_annot"].to(config.General.device)]
        )
        return {"classification": classification, "regression": regression}
    if args.network_type == "counting_lean":
        l1_loss, maps = model([image, data["annot"]])
        return {"l1_estimation": l1_loss, "maps": maps}
    if args.network_type == "counting_reg":
        return {"reg_estimation": model([image, data["annot"]])}
    if not torch.cuda.is_available():
        raise RuntimeError(f"Network type {args.network_type} requires CUDA for training.")

    points = data.get("points_annot")
    annotations = [data["bbox_annot"].to(config.General.device), points]
    group_tensor = torch.tensor(sampler_group)
    if args.network_type == "both":
        if config.AttributeEstimation.estimate_type == "withKeyPoints":
            classification, regression, counting, maps = model(
                [image, annotations, group_tensor]
            )
            return {
                "classification": classification,
                "regression": regression,
                "l1_counting": counting,
                "maps": maps,
            }
        classification, regression, counting = model(
            [image, annotations, group_tensor]
        )
        return {
            "classification": classification,
            "regression": regression,
            "counting": counting,
        }

    model_input = [image, annotations, group_tensor]
    if config.AttributeEstimation.estimate_type == "withKeyPoints":
        classification, regression, color, maps, length, diameter = model(model_input)
        return {
            "classification": classification,
            "regression": regression,
            "color": color,
            "maps": maps,
            "length": length,
            "diameter": diameter,
        }
    model_input.append(args.do_counting)
    classification, regression, color, length, diameter = model(model_input)
    return {
        "classification": classification,
        "regression": regression,
        "color": color,
        "length": length,
        "diameter": diameter,
    }


def run_training_step(
    model: Any,
    optimizer: Any,
    data: Dict[str, Any],
    args: Any,
    sampler_group: Any,
) -> Optional[LossResult]:
    """Execute forward, backward, gradient clipping, and optimization for one batch."""
    optimizer.zero_grad()
    result = combine_losses(forward_losses(model, data, args, sampler_group), args)
    if result.total is None or bool(result.total == 0):
        return None
    if result.total.grad_fn is not None:
        result.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    return result
