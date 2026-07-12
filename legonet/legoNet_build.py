"""Model registry for the active LegoNet construction flow.

The public build path is intentionally centralized here so that model
selection, imports, and constructor wiring live in one place. Model
composition stays inside the individual model modules, which in turn reuse the
shared lego blocks from ``legos.py``.
"""

from importlib import import_module
import config

def _build_bbox_detection(module, args, dataset):
    return module.BBOX_Detection(num_classes=dataset.num_classes())


def _build_per_object_estimate(module, args, dataset):
    return module.PerObjectEstimate(
        dataset=dataset,
        network_type=args.network_type,
        num_classes=dataset.num_classes(),
        freeze_detection=args.freeze_detection,
    )


def _build_KP_estimator(module, args, dataset):
    return module.ImageEstimatorWithKeypoints(
        dataset=dataset,
        network_type=args.network_type,
        num_classes=dataset.num_classes(),
    )


def _build_reg_estimator(module, args, dataset):
    return module.ImageEstimatorWithReg(
        dataset=dataset,
        network_type=args.network_type,
        num_classes=dataset.num_classes(),
    )


MODEL_REGISTRY = {
    "bbox_detection": {
        "module_path": "legonet.models.model_bbox_detection",
        "builder": _build_bbox_detection,
    },
    "per_image_estimation_keypoints": {
        "module_path": "legonet.models.model_estimator_withKP",
        "builder": _build_KP_estimator,
    },
    "per_image_estimation_regression": {
        "module_path": "legonet.models.model_estimator_withReg",
        "builder": _build_reg_estimator,
    },
    "per_object_counting": {
        "module_path": "legonet.models.model_per_object_counting",
        "builder": _build_per_object_estimate,
    },
    "per_object_attributes": {
        "module_path": "legonet.models.model_per_object_attributes",
        "builder": _build_per_object_estimate,
    },
    "per_object_attributes_multibranch":{
        "module_path": "legonet.models.model_per_object_attributes_multibranch",
    "builder": _build_per_object_estimate,}
}


def model_build(args, dataset_train, dataset_val):
    """Instantiate the active model variant for the current run arguments."""
    dataset = dataset_train if args.run_script == "Training" else dataset_val

    if args.network_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model variant: {args.network_type}")

    if args.network_type == "per_object_attributes_multibranch":
        assert config.AttributeEstimation.estimate_type == 'withKeyPoints'

    model_config = MODEL_REGISTRY[args.network_type]
    model_module = import_module(model_config["module_path"])
    return model_config["builder"](model_module, args, dataset)
