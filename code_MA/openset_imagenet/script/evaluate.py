""" Independent code for inference in testing dataset."""
import argparse
import os, sys
from pathlib import Path
import numpy as np
import torch
from vast.tools import set_device_gpu, set_device_cpu, device
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import openset_imagenet
import pickle
from openset_imagenet.openmax_evm import compute_adjust_probs, compute_probs, get_param_string
from loguru import logger

def command_line_options(command_line_arguments=None):
    """Gets the evaluation parameters."""
    parser = argparse.ArgumentParser("Get parameters for evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # directory parameters
    parser.add_argument(
        "--losses", "-l",
        choices = ["entropic", "softmax", "garbage"],
        nargs="+",
        default = ["entropic", "softmax", "garbage"],
        help="Which loss functions to evaluate"
    )
    parser.add_argument(
        "--protocols", "-p",
        type = int,
        nargs = "+",
        choices = (1,2,3),
        default = (2,1,3),
        help = "Which protocols to evaluate"
    )
    parser.add_argument(
        "--algorithms", "-a",
        choices = ["threshold", "openmax", "proser", "evm"],
        nargs = "+",
        default = ["threshold", "openmax", "proser", "evm"],
        help = "Which algorithm to evaluate. Specific parameters should be in the yaml file"
    )

    parser.add_argument(
        "--configuration", "-c",
        type = Path,
        default = Path("config/test.yaml"),
        help = "The configuration file that defines the experiment"
    )

    parser.add_argument(
        "--use-best", "-b",
        action="store_true",
        help = "If selected, the best model is selected from the validation set. Otherwise, the last model is used"
    )

    parser.add_argument(
        "--gpu", "-g",
        type = int,
        nargs="?",
        default = None,
        const = 0,
        help = "Select the GPU index that you have. You can specify an index or not. If not, 0 is assumed. If not selected, we will train on CPU only (not recommended)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help = "If selected, all results will always be recomputed. If not selected, already existing results will be skipped."
    )

    args = parser.parse_args(command_line_arguments)
    return args

def dataset(cfg, protocol, cfg_ns):
    # Create transformations
    transform = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()])

    # We only need test data here, since we assume that parameters have been selected
    if(cfg_ns.evaluate == 'val'): 
        test_dataset = openset_imagenet.ImagenetDataset(
            csv_file=cfg.data.val_file.format(protocol),
            imagenet_path=cfg.data.imagenet_path,
            transform=transform)
    else:
        test_dataset = openset_imagenet.ImagenetDataset(
            csv_file=cfg.data.test_file.format(protocol),
            imagenet_path=cfg.data.imagenet_path,
            transform=transform)

    # Info on console
    logger.info(f"Loaded test dataset for protocol {protocol} with len:{len(test_dataset)}, labels:{test_dataset.label_count}")

    # create data loaders
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers)

    # return test loader
    return test_dataset, test_loader


def load_model(cfg, loss, algorithm, protocol, suffix, output_directory, n_classes,cfg_ns):
    model = openset_imagenet.ResNet50Layers(
            fc_layer_dim=n_classes,
            out_features=n_classes,
            logit_bias=False)

    model_path = cfg.model_path.format(output_directory, loss, "threshold", suffix, cfg_ns.experiment_number)


    if not os.path.exists(model_path):
        logger.warning(f"Could not load model file {model_path}; skipping")
        return

    start_epoch, best_score = openset_imagenet.train.load_checkpoint(model, model_path)
    logger.info(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
    device(model)
    return model


def extract(model, data_loader, algorithm, loss):
    return openset_imagenet.train.get_arrays(
            model=model,
            loader=data_loader,
            garbage=loss=="garbage",
            pretty=True
        )


def post_process(gt, logits, features, scores, cfg, protocol, loss, algorithm, output_directory, gpu):

    if algorithm in ("threshold", "proser"):
        return scores

    opt = cfg.optimized[algorithm]
    popt = opt[f"p{protocol}"][loss]

    if algorithm == "openmax":
        key = get_param_string("openmax", tailsize=popt.tailsize, distance_multiplier=popt.distance_multiplier)
    else:
        key = get_param_string("evm", tailsize=popt.tailsize, distance_multiplier=popt.distance_multiplier, cover_threshold=opt.cover_threshold)
    model_path = opt.output_model_path.format(output_directory, loss, algorithm, key, opt.distance_metric)
    if not os.path.exists(model_path):
        logger.warning(f"Could not load model file {model_path}; skipping")
        return
    model_dict = pickle.load(open(model_path, "rb"))

    hyperparams = openset_imagenet.util.NameSpace(dict(distance_metric = opt.distance_metric))
    if algorithm == 'openmax':
        #scores are being adjusted her through openmax alpha
        logger.info("adjusting probabilities for openmax with alpha")
        return compute_adjust_probs(gt, logits, features, scores, model_dict, "openmax", gpu, hyperparams, popt.alpha)
    elif algorithm == 'evm':
        logger.info("computing probabilities for evm")
        return compute_probs(gt, logits, features, scores, model_dict, "evm", gpu, hyperparams)

def write_scores(gt, logits, features, scores, loss, algorithm, suffix, output_directory, cfg_ns):
    file_path = Path(output_directory) / f"{loss}_{algorithm}_{cfg_ns.evaluate}_arr_{suffix}_{str(cfg_ns.experiment_number)}.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    logger.info(f"Target labels, logits, features and scores saved in: {file_path}")

def load_scores(loss, algorithm, suffix, output_directory, cfg_ns):
    file_path = Path(output_directory) / f"{loss}_{algorithm}_{cfg_ns.evaluate}_arr_{suffix}_{str(cfg_ns.experiment_number)}.npz"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        return None

    
def process_model(protocol, loss, algorithm, cfg, cfg_ns, suffix, gpu, force):
    output_directory = Path(cfg.output_directory)/f"Protocol_{protocol}"/f"{cfg_ns.method}"

    # set device
    if gpu is not None:
        set_device_gpu(index=gpu)
    else:
        logger.warning("No GPU device selected, evaluation will be slow")
        set_device_cpu()

    # get dataset
    test_dataset, test_loader = dataset(cfg, protocol, cfg_ns)

    # load base model
    if loss == "garbage":
        n_classes = test_dataset.label_count - 1 # we use one class for the negatives; the dataset has two additional  labels: negative and unknown
    else:
        n_classes = test_dataset.label_count - 2  # number of classes - 2 when training was without garbage class
        if(cfg_ns.evaluate == 'val'):
            n_classes = test_dataset.label_count - 1

    base_data = None if force else load_scores(loss, algorithm, suffix, output_directory,cfg_ns)
    if base_data is None:
        logger.info(f"Loading base model for protocol {protocol}, {loss}")
        base_model = load_model(cfg, loss, algorithm, protocol, suffix, output_directory, n_classes,cfg_ns)
        if base_model is not None:
             # extract features
            logger.info(f"Extracting base scores for protocol {protocol}, {loss}")
            gt, logits, features, base_scores = extract(base_model, test_loader, algorithm, loss)
            write_scores(gt, logits, features, base_scores, loss, algorithm, suffix, output_directory,cfg_ns)
            # remove model from GPU memory
            del base_model
    else:
        logger.info("Using previously computed features")
        gt, logits, features, base_scores = base_data["gt"], base_data["logits"], base_data["features"], base_data["scores"]

    if algorithm not in ("proser", "threshold"):
            logger.info(f"Post-processing scores for protocol {protocol}, {loss} with {algorithm}")
            # post-process scores
            scores = post_process(gt, logits, features, base_scores, cfg, protocol, loss, algorithm, output_directory, gpu)
            if scores is not None:
                write_scores(gt, logits, features, scores, loss, algorithm, suffix, output_directory,cfg_ns)

def main(command_line_arguments = None):

    args = command_line_options(command_line_arguments)
    cfg = openset_imagenet.util.load_yaml(args.configuration)
    cfg_ns = openset_imagenet.util.load_yaml('config/neg_samples.yaml')
    suffix = "best" if args.use_best else "curr"
    protocol = args.protocols[0]
    loss = args.losses[0]

    print(protocol)

    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
    logger.add(
        sink= Path(cfg.output_directory) / cfg.log_name,
        format=msg_format,
        level="INFO",
        mode='w')

    process_model(protocol, loss, args.algorithms[0], cfg, cfg_ns, suffix, args.gpu, args.force)

if __name__=='__main__':
    main()