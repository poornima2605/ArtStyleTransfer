import torch
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torchvision.models import vgg19
from torch.optim import Adam

from PIL import Image, ImageFilter


def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


def run_style_transfer(content, style, vgg, content_weight=1e3,
                       style_weight=1e3, steps=1000):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    content = content.to(device)
    style = style.to(device)

    # Generate initial target image
    target = content.clone().requires_grad_(True).to(device)

    # Define optimizer
    optimizer = Adam([target], lr=0.03)

    # Define loss function layers
    def get_features(image, model, layers=None):
        if layers is None:
            layers = {
                '0': 'conv1_1',
                '5': 'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
                '21': 'conv4_2',  # Content Layer
                '28': 'conv5_1'
            }
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # Calculate gram matrix for style loss
    def gram_matrix(tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in
                   style_features}

    for step in range(steps):
        target_features = get_features(target, vgg)

        # Content loss
        content_loss = torch.mean(
            (target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # Style loss
        style_loss = 0
        for layer in style_features:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (target_feature.shape[1] ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, Total Loss: {total_loss.item()}")
    return target


def perform_style_transfer(content_path, style_path, output_path):
    """
    Perform style transfer on the content and style images and save the output.
    Args:
        content_path (str): Path to the content image.
        style_path (str): Path to the style image.
        output_path (str): Path to save the processed image.
    """
    # Load images
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # Load Pre-trained VGG19 Model
    vgg = vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    vgg.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Perform the style transfer (using your existing logic)
    output_image = run_style_transfer(content_image, style_image, vgg)

    image = output_image.cpu().detach().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    image_ori = image * std[:, None, None] + mean[:, None, None]
    image_ori1 = transforms.functional.to_pil_image(image_ori)
    image_ori1 = image_ori1.convert('RGB')

    filtered_image = image_ori1.filter(ImageFilter.MedianFilter(size=5))
    filtered_image = filtered_image.filter(
        ImageFilter.GaussianBlur(radius=0.5))

    # Save the result
    filtered_image.save(output_path)
    print("Stylized image saved!")

content_image_path = "./InputImage/vase.jpg"
style_image_path = "./InputImage/greenforest.jpg"
perform_style_transfer(content_image_path, style_image_path,
                       "./style/ProcessedImage/output.jpg")
