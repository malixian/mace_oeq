import argparse
import logging
import os
from typing import Dict, List, Tuple

import torch

from mace.tools.scripts_utils import extract_config_mace_model



def run(
    input_model,
    output_model="_oeq.model",
    device="cpu",
    return_model=True,
):
    # Setup logging

    # Load original model
    # logging.warning(f"Loading model")
    # check if input_model is a path or a model
    if isinstance(input_model, str):
        source_model = torch.load(input_model, map_location=device)
    else:
        source_model = input_model
    default_dtype = next(source_model.parameters()).dtype
    torch.set_default_dtype(default_dtype)
    # Extract configuration
    config = extract_config_mace_model(source_model)
    config["oeq_config"] = {"enabled": True, "conv_fusion": "deterministic"}

    # Create new model with openequivariance config
    logging.info("Creating new model with openequivariance settings")
    target_model = source_model.__class__(**config).to(device)
    
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    # To migrate openequivariance, we should transfer all keys
    for key in target_dict:
        if key in source_dict:
            target_dict[key] = source_dict[key]

    target_model.load_state_dict(target_dict)

    # Transfer weights with proper remapping
    # transfer_weights(source_model, target_model, max_L, correlation)

    if return_model:
        return target_model

    if isinstance(input_model, str):
        base = os.path.splitext(input_model)[0]
        output_model = f"{base}.{output_model}"
    logging.warning(f"Saving CuEq model to {output_model}")
    torch.save(target_model, output_model)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Path to input MACE model")
    parser.add_argument(
        "--output_model",
        help="Path to output openequivariance model",
        default="oeq_model.pt",
    )
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument(
        "--return_model",
        action="store_false",
        help="Return model instead of saving to file",
    )
    args = parser.parse_args()

    run(
        input_model=args.input_model,
        output_model=args.output_model,
        device=args.device,
        return_model=args.return_model,
    )


if __name__ == "__main__":
    main()
