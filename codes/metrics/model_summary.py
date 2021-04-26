import torch
import torch.nn as nn


# define modules to be incorporated in FLOPs computation
registered_module = [
    nn.Conv2d,
    nn.ConvTranspose2d
]

# initialize
registered_hooks, model_info_lst = [], []


def calc_gflops_per_batch(module, out_h, out_w):
    """ Calculate flops of conv weights (support groups_conv & dilated_conv)
    """
    gflops = 0
    if hasattr(module, 'weight'):
        # Note: in_c is already divided by groups while out_c is not
        out_c, in_c, k_h, k_w = module.weight.shape
        gflops += 2*in_c*k_h*k_w*out_c*out_h*out_w/1e9
    return gflops


def hook_fn_forward(module, input, output):
    batch_size, _, out_h, out_w = output.size()
    gflops = batch_size * calc_gflops_per_batch(module, out_h, out_w)
    model_info_lst.append({'gflops': gflops})


def register_hook(module):
    if isinstance(module, tuple(registered_module)):
        registered_hooks.append(module.register_forward_hook(hook_fn_forward))


def register(model, dummy_input_dict):
    # reset params
    global registered_hooks, model_info_lst
    registered_hooks, model_info_lst = [], []

    # register hook
    model.apply(register_hook)

    # forward
    _ = model(**dummy_input_dict)

    # remove hooks
    for hook in registered_hooks:
        hook.remove()


def profile_model(model):
    tot_gflops = 0
    for module_info in model_info_lst:
        if module_info['gflops']:
            tot_gflops += module_info['gflops']

    tot_params = 0
    for param in model.parameters():
        tot_params += torch.prod(torch.tensor(param.size())).item()

    return tot_gflops, tot_params
