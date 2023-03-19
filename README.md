# NTIRE2023-ESR-RepRFN
Code, pre-train model and factsheet of RepRFN for NTIRE 2023 Challenge on Efficient Super-Resolution. Our team ID is 7.

## How to test the model?
1. `git clone https://github.com/laonafahaodange/NTIRE2023-ESR-RepRFN.git`
2. Set the `--model_id 0` / `--model_id 7` to test the baseline / RepRFN model from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 0
    ```
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 7
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.

## How to calculate the number of parameters, FLOPs, and activations
Code similar to the following is used.
```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team00_RFDN import RFDN
    model = RFDN()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```
**Specifically you can run the `models/team07_RepRFN.py` to get the result, remember to uncomment the code.**

## Thank all organizers for their efforts!

