## Introduction
Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks

## Requirements
* Python == 3.9
* openprompt==1.0.1
* transformers==4.35.2

## In-context Learning Backdoor Attack

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml context_learning.py --model facebook/opt-1.3b
```

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml attack_clean_sentence.py --model facebook/opt-1.3b
```

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml attack_sentence.py --model facebook/opt-1.3b
```

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml attack_clean_prompt.py --model facebook/opt-1.3b
```

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml attack_prompt.py --model facebook/opt-1.3b
```

## Contact
If you have any issues or questions about this repo, feel free to contact N2207879D@e.ntu.edu.sg.
