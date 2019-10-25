import torch

from networks.deeplabv3plus import get_model
# from modules.encoder import get_model
from utils.GFLOPs import add_flops_counting_methods, flops_to_string, get_model_parameters_number

net = get_model().cuda()
batch = torch.FloatTensor(1, 3, 513, 513).cuda()
model = add_flops_counting_methods(net)
model.eval().start_flops_count()
out = model(batch)[0]

print(model)
print('Output shape: {}'.format(list(out.shape)))
print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
print('Params: ' + get_model_parameters_number(model))
