Benchmarked devices:
    
    Xiaomi Mi notabook air 13.3
        CPU: Intel Core i5 8250U
        RAM: 8GB
        OS: Windows 10 Home x64 21H1
        GPU: Nvidia MX150 2GB
    
    Lenovo IdeaPad 5
        CPU: AMD RYZEN 3 4300U with radeon graphics 2.7 GHz
        RAM: 8GB
        OS: Windows 10 Pro 21H1
        GPU: NO

    Jetson Nano 4GB
        CPU: ARM Cortex-A57 (quad-core) @ 1.43GHz
        RAM: 4GB 64-bit LPDDR4
        OS: Linux Ubuntu JetPack
        GPU: 128x Maxwell @ 921 MHz (472 GFLOPS)
        
    (not yet implemented)
    Raspberry pi 4B
        CPU: Broadcom BCM2711, quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz
        RAM: 4GB LPDDR4-3200
        OS: Linux Raspbian OS (bullseye)
        GPU: No

[PC-windows-GPU]
installation
    requirements: 
        nvidia gpu, cuda version>=11.3 driver version 496.49 
        webcamera or some sort or a video stream (ffmpeg etc)
    - git clone https://github.com/batterlake/ssd-benchmark
    - create python environment:
        python -m venv env
    - activate python env
        env/Scripts/Activate.ps1
    install torch'n'stuff:
    - pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    - pip install -r requirements.txt
    - copy model checkpoint (can be optained by training ssd model with pytorch-ssd submodule) to "model" folder 
    
execution:
    - activate environment:
        env/Scripts/Activate.ps1
    - python run_ssd_onnx_camera.py .\model\fruit\mb1-ssd-Epoch-99-loss-3.454.pth .\model\fruit\labels.txt cuda

[PC-windows-CPU]
installation:
    requirements: 
        webcamera or some sort or a video stream (ffmpeg etc)
    - git clone https://github.com/batterlake/ssd-benchmark
    - create python environment:
        python -m venv env
    - activate python env
        env/Scripts/Activate.ps1
    install torch'n'stuff:
    - pip install torch torchvision torchaudio
    - pip install -r requirements.txt
    - copy model checkpoint (can be optained by training ssd model with pytorch-ssd submodule) to "model" folder 

train model:
    https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md

execution:
    - activate environment:
        env/Scripts/Activate.ps1
    - python run_ssd_onnx_camera.py .\model\fruit\mb1-ssd-Epoch-99-loss-3.454.pth .\model\fruit\labels.txt cpu

[JETSON NANO]
installation
    - Follow steps explained in https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro
    - clone jetson-inference repo 
        git clone https://github.com/dusty-nv/jetson-inference
    - cd jetson-inference
    - run docker image on your device:
        docker/run.sh
        
execution
    - run docker image
        cd jetson-inference
        docker/run.sh
    - copy trained model to jetson-inference/python/trainings/detection/ssd/models
    - docker exec -it <jetson-inference container name> bash
    - cd jetson-inference/python/trainings/detection/ssd
    - run detectnet headless (without display rtp to pc)
        detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            csi://0 rtp://your-ip:desired-port
        this starts rtp stream to selected cliend and specific port
    - run detectnet with attached display:
        detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
          csi://0
    
    - view video stream on selected pc:
        ffplay -protocol_whitelist file,rtp,udp -i .\stream-jetson.sdp
        stream-jetson.sdp is a text file with contents:
        """
        SDP:
        c=IN IP4 <ip-of-streaming-device>
        m=video <selected-port> RTP/AVP 96
        a=rtpmap:96 H264/90000
        """


OPENVINO
Convert model

python openvino-convert-ir2eng.py