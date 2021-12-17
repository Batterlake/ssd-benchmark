import logging
import numpy as np
import logging

from openvino.inference_engine import IECore, IENetwork

from openvino_utils import (
    request_sync,
    timed
)


# @timed
# def optimize_network(config):
#     """
#     Output of this function is exportable executable network, optimized and prepared for inference
#     """
#     plugin = IEPlugin(device=config.device) #MYRIAD
#     net = IENetwork(**config.network_kwargs) #model=model_xml, weights=model_bin OR ONNX
#     exec_net = plugin.load(network=net, num_requests=config.num_requests) #1
#     return exec_net

@timed
def load_export_optimized_model(
    model: str,
    weights: str = None,
    path_to_save_intermediete: str = None,
    device: str = "MYRIAD",
    num_requests: int = 1
):
    ie = IECore()
    net = ie.read_network(model=model, weights=weights)
    exec_net = ie.load_network(
        network=net,
        device_name=device,
        num_requests=num_requests
    )
    exec_net.export(path_to_save_intermediete)
    logging.info(f"Exported model to: {path_to_save_intermediete}")
    logging.info(f"Device: {device}")
    logging.info(f"Number reqiests: {num_requests}")
    return exec_net


def main():
    model = load_export_optimized_model(
        model="models/fruit/ssd-mobilenet.onnx",
        num_requests=1,
        path_to_save_intermediete='models/vino/fruit/ssd-mobilenet.een')

    batch = np.random.uniform(0,1,size=[1,3,300,300])
    result = request_sync(model,batch)
    print(result.keys())
    print("Export done with success")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
