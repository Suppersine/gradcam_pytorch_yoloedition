import torch
import torch.nn.functional as F

from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer, find_yolo_layer, detupler


class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'yolo' in model_type.lower():
            target_layer = find_yolo_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])



    def forward(self, input, class_idx=None, mode ='1', retain_graph=False): #see mode selection below
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        
        # Case 1: Raw feature map (logit[0])
        score_rfm = logit[0].squeeze()
        
        # Case 2: Objectness Scores (Small, Medium, Large, Average)
        score_obj_small = logit[1][0][..., 4].squeeze()
        score_obj_medium = logit[1][1][..., 4].squeeze()
        score_obj_large = logit[1][2][..., 4].squeeze()
        print(logit[1][0].size())
        print(logit[1][1].size())
        print(logit[1][2].size())
        # logitbar = (logit[1][0] + logit[1][1] + logit[1][2]) / 3
        # score_obj_avg = logitbar[..., 4].squeeze()
        
        # Case 3: Class Probabilities (Small, Medium, Large, Average)
        score_prob_small = logit[1][0][..., 5:].squeeze()
        score_prob_medium = logit[1][1][..., 5:].squeeze()
        score_prob_large = logit[1][2][..., 5:].squeeze()
        # score_prob_avg = logitbar[..., 5:].squeeze()
        
        # Programically select a mode, 
        """mode = input("Enter a YOLO pixel-wise metric to evaluate:\n"
                     "1.rfm, 2.obj_small, 3.obj_medium, 4.obj_large,\n"
                     "5.prob_small, 6.prob_medium, 7.prob_large, 8.non-Yolo\n"
                     "Your choice: ").strip()"""

        #hardcoded for troubleshooting purposes
        mode = '2'
        #mode = '5'

        # Process the input to determine which score to use
        if mode == '1':
            score = score_rfm
        elif mode == '2':
            score = score_obj_small
        elif mode == '3':
            score = score_obj_medium
        elif mode == '4':
            score = score_obj_large
        elif mode == '5':
            #class_idx = int(input("Enter class index (or -1 for max class probability): ").strip())
            if class_idx is not None:
                score = score_prob_small[..., class_idx].squeeze()
            else:
                score = score_prob_small.max(dim=-1)[0].squeeze()
        elif mode == '6':
            #class_idx = int(input("Enter class index (or -1 for max class probability): ").strip())
            if class_idx is not None:
                score = score_prob_medium[..., class_idx].squeeze()
            else:
                score = score_prob_medium.max(dim=-1)[0].squeeze()
        elif mode == '7':
            #class_idx = int(input("Enter class index (or -1 for max class probability): ").strip())
            if class_idx is not None:
                score = score_prob_large[..., class_idx].squeeze()
            else:
                score = score_prob_large.max(dim=-1)[0].squeeze()
        elif mode == '8': #non-YOLO
            if class_idx is None:
                score = logit[:, logit.max(1)[-1]].squeeze()
            else:
                score = logit[:, class_idx].squeeze() 
        else:
            raise ValueError("Invalid choice! Please select a valid mode.")
        
        # Backpropagation and Grad-CAM computation
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        
        # Get gradients and activations
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()
        
        # Compute weights (global average pooling over gradients)
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        
        # Compute saliency map
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        
        # Normalize the saliency map
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        
        return saliency_map, logit


    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze() 
            
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit
