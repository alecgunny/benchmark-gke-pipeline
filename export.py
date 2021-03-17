import argparse
import typing

import tensorflow as tf
import torch

from deepclean_prod.nn.net import DeepClean
from mldet.net import Net as BBHNet

from exportlib import ModelRepository
from exportlib.platform import PlatformName


tf.config.set_visible_devices([], 'GPU')
BATCH_SIZE = 1


class PostProcessor(torch.nn.Module):
    def forward(self, strain, noise_h, noise_l):
        # TODO: needs to add:
        #    - filtering
        #    - de-centering
        #    - any preprocessing for bbh
        noise = torch.stack([noise_h, noise_l], dim=1)
        return strain - noise


def parse_platform(platform):
    # do some parsing of the deepclean export platform
    deepclean_export_kwargs = {
        "output_names": ["noise"]
    }
    url = None
    try:
        # see if we have a URL appended to it
        platform, url = platform.split(":", maxsplit=1)
    except ValueError:
        # nope, ok move on
        pass

    try:
        # see if we have the format trt_<precision>
        platform, precision = platform.split("_")
    except ValueError:
        # nope, ok this is (presumably) ONNX
        assert platform == "onnx"
    else:
        # indicate we should use fp16 if it was specified
        if precision == "fp16":
            deepclean_export_kwargs["use_fp16"] = True

        # add in the URL, which could just be None
        # don't add it for onnx since that platform
        # doesn't have that kwarg
        deepclean_export_kwargs["url"] = url

    # now map to a real Platform
    try:
        platform = PlatformName.__members__[platform.upper()].value
    except KeyError:
        raise ValueError(f"Unrecognized platform {platform}")
    return platform, deepclean_export_kwargs


def main(
    repo_dir: str,
    platform: str = "onnx",
    gpus: typing.Optional[int] = None,
    count: int = 1,
    base_name: typing.Optional[str] = None,
    kernel_stride: float = 0.002,
    fs: float = 4000,
    kernel_size: float = 1.0,
    streams_per_gpu: int = 1,
):
    repo = ModelRepository(repo_dir)
    snapshot_size = int(fs * kernel_size)
    base_name = base_name + "_" if base_name is not None else ""

    platform, deepclean_export_kwargs = parse_platform(platform)
    witness_channels = {"h": 21, "l": 21}
    deepcleans = {}
    for detector, num_channels in witness_channels.items():
        # basic formula:
        # 1: create the model architecture (i.e. the torch Module
        # that executes all of the operations that perform your
        # input-output mapping). This would typically be done
        # in order to perform training
        arch = DeepClean(num_channels)
        arch.eval()

        # 1b: train the model, which we're not doing here

        # 2: add a directory for the model architecture to
        # your model repository. This `Model` object represents
        # that repository entry and its associated metadata
        model = repo.create_model(
            f"{base_name}deepclean_{detector}", platform=platform
        )
        # 3: Add an instance group that tells Triton how many
        # GPUs to leverage, and how much to parallelize model
        # execution on each GPU
        model.config.add_instance_group(
            count=count,  # gpus=gpus, count=count
        )

        # 4: Export the current model architecture to the
        # model's entry in the repository. This will be creating
        # some binary representation that is consistent with the
        # model's metadata and that Triton can leverage to perform
        # execution at inference time (indeed, the export step
        # itself records much of the metadata, since it's here
        # that we freeze in things like input shapes)
        model.export_version(
            arch,
            input_shapes={"witness": (BATCH_SIZE, num_channels, snapshot_size)},
            **deepclean_export_kwargs
        )

        deepcleans[detector] = model

    postprocessor = PostProcessor()
    postprocessor.eval()

    pp_model = repo.create_model(
        f"{base_name}postproc", platform=PlatformName.ONNX
    )
    pp_model.config.add_instance_group(
        count=count  # , gpus=gpus
    )
    pp_model.export_version(
        postprocessor,
        input_shapes={
            "strain": (BATCH_SIZE, 2, snapshot_size),
            "noise_h": (BATCH_SIZE, snapshot_size),
            "noise_l": (BATCH_SIZE, snapshot_size)
        },
        output_names=["cleaned"]
    )

    bbh_params = {
        "filters": (3, 3, 3),
        "kernels": (8, 16, 32),
        "pooling": (4, 4, 4, 4),
        "dilations": (1, 1, 1, 1),
        "pooling_type": "max",
        "pooling_first": True,
        "bn": True,
        "linear": (64, 32),
        "dropout": 0.5,
        "weight_init": None,
        "bias_init": None
    }
    bbh = BBHNet((2, snapshot_size), bbh_params)
    bbh.eval()

    bbh_model = repo.create_model(
        f"{base_name}bbh", platform=PlatformName.ONNX
    )
    bbh_model.config.add_instance_group(
        count=count,  gpus=gpus
    )
    bbh_model.export_version(
        bbh,
        input_shapes={"strain": (BATCH_SIZE, 2, snapshot_size)},
        output_names=["prob"]
    )

    ensemble = repo.create_model(
        f"{base_name}gwe2e", platform=PlatformName.ENSEMBLE
    )
    ensemble.add_streaming_inputs(
        inputs=[
            deepcleans["h"].inputs["witness"],
            deepcleans["l"].inputs["witness"],
            pp_model.inputs["strain"]
        ],
        stream_size=int(kernel_stride * fs),
        name=f"{base_name}snapshotter",
        streams_per_gpu=streams_per_gpu
    )

    for detector, model in deepcleans.items():
        ensemble.pipe(
            model.outputs["noise"],
            pp_model.inputs[f"noise_{detector}"],
            name=f"noise_{detector}"
        )
        # ensemble.add_output(model.outputs["noise"], name=f"noise_{detector}")
    ensemble.pipe(
        pp_model.outputs["cleaned"],
        bbh_model.inputs["strain"]
    )
    ensemble.add_output(bbh_model.outputs["prob"])
    ensemble.export_version()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-dir",
        type=str,
        required=True,
        help="Path to save model repository to"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="onnx",
        help=(
            "Format to export deepclean models in. "
            "Choices are 'onnx', 'trt_fp32', or 'trt_fp16'. "
            "Additionally, TensorRT formats can be appended with "
            ":<url> to indicate that TRT conversion should take "
            "place on the indicated remote server. For example, "
            "'trt_fp16:http://localhost:5000/onnx' would indicate that "
            "there is a local TensorRT conversion service to perform "
            "conversion to an FP16 inference engine."
        )
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to host service on"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of model instances to place per GPU"
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default=None,
        help="Name to prepend to all models in the ensemble"
    )

    # snapshot parameters
    parser.add_argument(
        "--kernel-stride",
        type=float,
        default=0.002,
        help="Time between frame snapshots in seconds"
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=4000,
        help="Samples in a frame"
    )
    parser.add_argument(
        "--kernel-size",
        type=float,
        default=1.0,
        help="Length of snapshot windows in seconds"
    )
    parser.add_argument(
        "--streams-per-gpu",
        type=int,
        default=1,
        help="Number of snapshotter instances to host per GPU"
    )
    flags = parser.parse_args()
    main(**vars(flags))
