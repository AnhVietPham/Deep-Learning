import copy
import torchvision.models as models

#
# def get_model(model_dict, n_classes, version=None):
#     name = model_dict['arch']
#     model = _get_model_instance(name)
#     param_dict = copy.deepcopy(model_dict)
#     param_dict.pop("arch")
#
#     if name in ["fcn8s"]:
#         model = model(n_classes=n_classes, **param_dict)
#         vgg16 = models.vgg16(pretrained=True)
#         model.init_vgg16_params(vgg16)
#     return model


# def _get_model_instance(name):
#     try:
#         return {
#             "fcn8s": FCN8s
#         }[name]
#     except:
#         raise ("Model {} not available".format(name))
