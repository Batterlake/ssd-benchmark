import logging
import time
from openvino.inference_engine import IECore


def timed(funk):
    def wrapper(*args,**kwargs):
        print(f"{funk.__name__} begin")
        st_time = time.time()
        res = funk(*args,**kwargs)
        ex_time = time.time() - st_time
        print(f"{funk.__name__} end")
        print(f'Execution time: {ex_time}')
        return res

    return wrapper


@timed
def import_network(path, device='MYRIAD', num_requests=1):
    ie = IECore()
    """
    Import previously exported model. Path is .een file
    """
    exec_net_imported = ie.import_network(model_file=path, device_name=device,num_requests=num_requests)
    logging.info("Imported model")
    logging.info(f"Device: {device}")
    logging.info(f"Num requests: {num_requests}")
    return exec_net_imported


def request_sync(executable_model, input_batch):
    inoput_blob_name = next(iter(executable_model.inputs))
    return executable_model.infer(inputs={inoput_blob_name: input_batch})