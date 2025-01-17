from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor
import cv2
import numpy as np
import torch

def detuple(data):
    """
    Recursively detuples nested tuples until a tensor or a list of tensors is found.
    """
    if isinstance(data, tuple):
        if all(isinstance(x, torch.Tensor) for x in data):
            # If all elements are tensors, return the first one.
            # Modify this if you need a different element from the tuple.
            return data[0]
        else:
            # Recursively detuple nested tuples
            return [detuple(x) for x in data]
    elif isinstance(data, list):
        # If it's a list, recursively detuple its elements.
        return [detuple(x) for x in data]
    else:
        # If it's not a tuple or a list, return as is.
        return data

def logitprocessor(logit, yolomode, class_idx = None):
    
    # Programmatically select score based on yolomode
    if yolomode == '1': # HBB & OBB Case 1: Backbone Raw Feature Map
        score_rfm = logit[0].squeeze().mean()  # Global average
        score = score_rfm
    elif yolomode == '2': # HBB Case 2: Objectness / Output format: logit[1] = [sz x sz]*[x, y ,w, h, obj_clfloat, clprob * nclass], where sz = 28
        score_obj_large = logit[1][0][..., 4].squeeze().max()  # Max objectness
        score = score_obj_large
    elif yolomode == '3': # HBB Case 3: Objectness / Output format: logit[1] = [sz x sz]*[x, y ,w, h, obj_clfloat, clprob * nclass], where sz = 14
        score_obj_medium = logit[1][1][..., 4].squeeze().max() # Max objectness
        score = score_obj_medium
    elif yolomode == '4': # HBB Case 4: Objectness / Output format: logit[1] = [sz x sz]*[x, y ,w, h, obj_clfloat, clprob * nclass], where sz = 7, which correlates to object sizes
        score_obj_small = logit[1][2][..., 4].squeeze().max() # Max objectness
        score = score_obj_small
    elif yolomode == '5': # HBB Case 5: Class Probabilities - Large Filter 28*28 (for large objects)
        if class_idx is not None: # For a specific class
            score_prob_large = logit[1][0][..., 5:].squeeze()[..., class_idx].max()
        else: # Max probability across all classes
            score_prob_large = logit[1][0][..., 5:].squeeze().max()
        score = score_prob_large
    elif yolomode == '6': # HBB Case 6: Class Probabilities - Medium Filter 14*14 (for medium objects)
        if class_idx is not None: # For a specific class
            score_prob_medium = logit[1][1][..., 5:].squeeze()[..., class_idx].max()
        else: # Max probability across all classes
            score_prob_medium = logit[1][1][..., 5:].squeeze().max()
        score = score_prob_medium
    elif yolomode == '7': # HBB Case 7: Class Probabilities - Small Filter 7*7 (for small objects)
        if class_idx is not None: # For a specific class
            score_prob_small = logit[1][2][..., 5:].squeeze()[..., class_idx].max()
        else: # Max probability across all classes
            score_prob_small = logit[1][2][..., 5:].squeeze().max()
        score = score_prob_small
    elif yolomode == '8':  # Case 8: non-YOLO models
        if class_idx is None:
            score_nonyolo = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score_nonyolo = logit[:, class_idx].squeeze()
        score = score_nonyolo
    elif yolomode == '9': # Case 9: OBB Detection Head Raw Feature Map
        score_rfm = logit[1][1].squeeze().mean()  # Global average
        score = score_rfm
    elif yolomode == '10': #obb-obj Output format: logit[1][0][sz] = [sz x sz]*[x, y ,w, h, theta, obj_clfloat, clprob * nclass], where sz = 28
        score_obj_large = logit[1][0][0][..., 5].squeeze().max()  # Max objectness
        score = score_obj_large
    elif yolomode == '11': #obb-obj Output format: logit[1][0][sz] = [sz x sz]*[x, y ,w, h, theta, obj_clfloat, clprob * nclass], where sz = 14
        score_obj_medium = logit[1][0][1][..., 5].squeeze().max()  # Max objectness
        score = score_obj_medium
    elif yolomode == '12': #obb-obj Output format: logit[1][0][sz] = [sz x sz]*[x, y ,w, h, theta, obj_clfloat, clprob * nclass], where sz = 7
        score_obj_small = logit[1][0][2][..., 5].squeeze().max()  # Max objectness
        score = score_obj_small
    elif yolomode == '13': # Class Prob for [28*28] OBB filter
        if class_idx is not None: # For a specific class
            score_prob_large = logit[1][0][0][..., 6:].squeeze()[..., class_idx].max()
        else: # Max probability across all classes
            score_prob_large = logit[1][0][0][..., 6:].squeeze().max()
        score = score_prob_large
    elif yolomode == '14': # Class Prob for [14*14] OBB filter
        if class_idx is not None: # For a specific class
            score_prob_medium = logit[1][0][1][..., 6:].squeeze()[..., class_idx].max()
        else: # Max probability across all classes
            score_prob_medium = logit[1][0][1][..., 6:].squeeze().max()
        score = score_prob_medium
    elif yolomode == '15': # Class Prob for [7*7] OBB filter
        if class_idx is not None: # For a specific class
            score_prob_small = logit[1][0][2][..., 6:].squeeze()[..., class_idx].max()
        else: # Max probability across all classes
            score_prob_small = logit[1][0][2][..., 6:].squeeze().max()
        score = score_prob_small
    
    else:
        raise ValueError("Invalid mode! Choose a valid mode.")

    return score

def visualize_cam(mask, img, captions):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    # Create a black background for the caption banner
    banner_width = img.shape[3]  # Use image width for banner width
    banner_height = img.shape[2]  # Adjust banner height as needed
    banner = Image.new('RGB', (banner_width, banner_height), (0, 0, 0))  # Black background

    # Draw captions on the banner
    draw = ImageDraw.Draw(banner)
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Replace with your font path if needed
    except OSError:
        font = ImageFont.load_default()  # Use default font if arial.ttf not found
    draw.text((10, 10), captions, (255, 255, 255), font=font)  # White text

    # Convert banner to tensor
    banner_array = np.array(banner)
    banner_tensor = torch.from_numpy(banner_array).permute(2, 0, 1).float().div(255)

    print(heatmap.size())
    print(banner_tensor.size())
    
    return heatmap, result, banner_tensor


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]
                
        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):
    """Find vgg layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_42'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features_12'
            target_layer_name = 'features_12_expand3x3'
            target_layer_name = 'features_12_expand3x3_activation'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2]+'_'+hierarchy[3]]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


def find_yolo_layer(model_arch, layer_name):
    """
    Locate the specified layer in the YOLO model architecture.

    Args:
        model_arch (torch.nn.Module): The YOLO model architecture.
        layer_name (str): The name of the layer to locate (e.g., 'model.23' or 'backbone.0').

    Returns:
        target_layer (torch.nn.Module): The requested layer/module.

    Raises:
        ValueError: If the specified layer is not found.
    """
    # Split the layer_name into components to handle hierarchical access (e.g., 'model.23' -> ['model', '23'])
    hierarchy = layer_name.split('.')

    # Start with the top-level model architecture
    target_layer = model_arch

    try:
        for part in hierarchy:
            if part.isdigit():  # If the part is a digit, it's an index for Sequential layers
                target_layer = target_layer[int(part)]  # Access the layer by index
            else:  # If the part is a string, use getattr to access the attribute (e.g., 'model', 'backbone')
                target_layer = getattr(target_layer, part)
    except (KeyError, IndexError, AttributeError) as e:
        raise ValueError(f"Layer '{layer_name}' not found in YOLO model: {e}")

    return target_layer

    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)
    
    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
