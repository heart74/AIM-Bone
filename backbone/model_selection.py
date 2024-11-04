"""
Author: Andreas Rössler
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import os

# from nbhh
class Classifier(nn.Module):
    def __init__(self,
                 encoder,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Classifier, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        binary = encoder.pop('type')
        self.net = binary_lib[binary](**encoder)
        self.cls_loss = nn.CrossEntropyLoss()
        self.init_weights(pretrained)

    def init_weights(self, pretrained):
        if pretrained is not None:
            state_dict = torch.load(pretrained)
            if 'last_linear.weight' in state_dict:
                state_dict.pop('last_linear.weight')
            if 'last_linear.bias' in state_dict:
                state_dict.pop('last_linear.bias')
            if 'fc.weight' in state_dict:
                state_dict.pop('fc.weight')
            if 'fc.bias' in state_dict:
                state_dict.pop('fc.bias')
            self.net.load_state_dict(state_dict, strict=False)

    def forward(self, img):
        return self.net(img)


# for Xception
def return_pytorch04_xception(pretrained=False):
    # Raises warning "src not broadcastable to dst" but thats fine
    from .xception import xception
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        pretrained_dict = torch.load( os.path.join(os.path.dirname(os.path.dirname(__file__)),'pretrained/Xecption_AUC99.69_Iter_020000.ckpt') )
        model_dict = model.state_dict()

        # for name, weights in pretrained_dict.items():
        #     if 'pointwise' in name:
        #         pretrained_dict[name] = weights.unsqueeze(-1).unsqueeze(-1) 

        #将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        # 更新现有的model_dict 
        model_dict.update(pretrained_dict)
        # 加载真正需要的state_dict 
        model.load_state_dict(model_dict)
        model.last_linear = model.fc
        del model.fc
    return model

# for Xception
class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


# backbone替换
# To be implemented: Xception skresnet18 resnest14d hrnet_w18_small_v2 IR_SE_18 TransFAS LGSC
def model_selection(modelname, num_out_classes, device="cuda:0", weight_path=None, use_fc=True):
    """
    :param modelname:
    :return: model
    """
    if modelname == 'mesonet':
        from .mesonet import MesoNet
        return MesoNet(num_class=num_out_classes), 256, True, ['image'], None
    elif modelname == 'xception':
        from torch.utils import model_zoo
        from collections import OrderedDict
        model = TransferModel(modelchoice='xception', num_out_classes=num_out_classes)
        # state_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth')
        weight_path = './model_zoo/xception-43020ad28.pth'
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'),)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'fc' in k:
                continue
            name = 'model.'+k # module字段在最前面，从第7个字符开始就可以去掉module
            new_state_dict[name] = v # 新字典的key值对应的value一一对应 
        model.load_state_dict(new_state_dict, strict=False)
        return model, 299, True, ['image'], None
    elif modelname == 'efficientnet-b0':
        from .efficientnet import EfficientNet
        original_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)),'model_zoo/efficientnet-b0-355c32eb.pth')
        if weight_path is None:
            # trained_weights  = os.path.join(os.path.dirname(os.path.dirname(__file__)),'model_zoo/efficientnet-b0-355c32eb.pth')
            # model = EfficientNet.from_pretrained('efficientnet-b0', weights_path=original_weights, num_classes=2)
            original_weights = weight_path
            model = EfficientNet.from_pretrained('efficientnet-b0',weights_path=original_weights, num_classes=2, use_fc=use_fc)
        else:
            model = EfficientNet.from_name('efficientnet-b0', num_classes=2, use_fc=use_fc)
            dic = torch.load(weight_path, map_location=torch.device('cpu'))
            if 'net_state_dict' in dic:
                model.load_state_dict(dic['net_state_dict'])
            else:
                model.load_state_dict(dic)
        # trained_weights  = os.path.join(os.path.dirname(os.path.dirname(__file__)),'/media/sda/hjr/workspace/dfdcv319/model/eff_lby_ssapi0.2_even_ss_sharpen_randomerase_2gpu_20210414_175946/AUC95.37_Iter_008000.ckpt')
        # trained_weights = '/media/sda/hjr/workspace/dfdcv319/model/jpgresv3.3_crop_aug_fwa_effb0_lr1e-4_20210416_002621/AUC98.18_Iter_007500.ckpt'
        print(original_weights)
        return model, 224, True, ['image'], None
    elif modelname == 'efficientnet-b7':
        from .efficientnet import EfficientNet
        # trained_weights  = os.path.join(os.path.dirname(os.path.dirname(__file__)),'pretrained/jpgresv3_crop_aug_fwa_effb0_auc93.21_iter7.5k.ckpt') #
        # trained_weights  = os.path.join(os.path.dirname(os.path.dirname(__file__)),'pretrained/effb2_AUC95.58_Iter_010000.ckpt')
        # trained_weights = '/media/sda/hjr/workspace/dfdcv319/model/effb2_lby_ssapi0.2_even_ss_sharpen_randomerase_2gpu_20210414_230450/AUC94.85_Iter_008000.ckpt'
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2, use_fc=use_fc)
        return model, 600, True, ['image'], None
    elif modelname == 'efficientnet-video':
        from .efficientnet import EfficientNet
        from .efficientnet_video import efficient_video
        original_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)),'model_zoo/efficientnet-b0-355c32eb.pth')
        if weight_path is None:
            original_weights = weight_path
            base_model = EfficientNet.from_pretrained('efficientnet-b0',weights_path=original_weights, num_classes=2, use_fc=use_fc)
            model = efficient_video(base_model, 1280, 2)
        else:
            base_model = EfficientNet.from_name('efficientnet-b0', num_classes=2, use_fc=use_fc)
            model = efficient_video(base_model, 1280, 2)
            model.load_state_dict(
                torch.load(weight_path, map_location=torch.device('cpu'))['net_state_dict']
            )
            print("loading efficient video weight")
        return model, 224, True, ['image'], None
    elif modelname == 'efficientnet-b2':
        from .efficientnet import EfficientNet
        # trained_weights  = os.path.join(os.path.dirname(os.path.dirname(__file__)),'pretrained/jpgresv3_crop_aug_fwa_effb0_auc93.21_iter7.5k.ckpt') #
        # trained_weights  = os.path.join(os.path.dirname(os.path.dirname(__file__)),'pretrained/effb2_AUC95.58_Iter_010000.ckpt')
        # trained_weights = '/media/sda/hjr/workspace/dfdcv319/model/effb2_lby_ssapi0.2_even_ss_sharpen_randomerase_2gpu_20210414_230450/AUC94.85_Iter_008000.ckpt'
        model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=2, use_fc=use_fc)
        return model, 260, True, ['image'], None
    elif modelname == 'resnest-14d':
        
        model = Classifier(
            encoder=dict(type='resnest14d', pretrained=False, num_classes=2))
        model_path = 'model_zoo/rs_937.pth'
        checkpoint = torch.load(model_path)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
            new_state_dict[name] = v # 新字典的key值对应的value一一对应 
        model.load_state_dict(new_state_dict)
        return model, 256, True, ['image'], None
    elif modelname == 'hrnet_w18_small_v2':
        import timm
        model = timm.create_model('hrnet_w18_small_v2', pretrained=False, num_classes=2)
        checkpoint = torch.load('model_zoo/hr_917.pth')
        model.load_state_dict(checkpoint)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
        #     new_state_dict[name] = v # 新字典的key值对应的value一一对应 
        # model.load_state_dict(new_state_dict)
        return model, 256, True, ['image'], None
    elif modelname == 'skresnet-18':      
        model = Classifier(
            encoder=dict(type='skresnet18', pretrained=False, num_classes=2))
        model_path = 'model_zoo/skr_923.pth'
        checkpoint = torch.load(model_path)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
            new_state_dict[name] = v # 新字典的key值对应的value一一对应 
        model.load_state_dict(new_state_dict)
        return model, 256, True, ['image'], None
    elif modelname == 'IResnet_SE18':
        from .iresnet import IR_SE_18
        model = IR_SE_18(num_classes=2, input_size=[224, 224])
        checkpoint = torch.load('./model_zoo/ISE_939.pth')['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
            new_state_dict[name] = v # 新字典的key值对应的value一一对应 
        model.load_state_dict(new_state_dict)
        return model, 224, True, ['image'], None
    # elif modelname == 'skresnet-18':
    #     raise NotImplementedError(modelname)

    else:
        # vit LGSC
        raise NotImplementedError(modelname)

# import os
# model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'pretrained/Xecption_AUC99.69_Iter_020000.ckpt')
# torch.load(model_path)