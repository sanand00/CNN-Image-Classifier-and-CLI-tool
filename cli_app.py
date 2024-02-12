import click
import torch
import os
from filesplit.merge import Merge
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as v2F

@click.command()
@click.option('--gpu', is_flag = True, default = False,
              help = 'Run CNN on an Nvidia GPU' ) 
@click.option('--path', prompt = 'Image Path',
              type=click.Path(exists=True),
              help = 'Path to an image of a cat or dog') 
def image_classify(gpu, path):
    try:
        img = Image.open(path, formats=('PNG','JPEG'))
    except TypeError:
        click.UsageError("Unsupported image file format, must be png or jpeg")
    
    device = 'cpu'
    
    if gpu:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            click.UsageError("Supported GPU not available.")
        
    img_tensor = v2F.pil_to_tensor(img.convert('RGB')).float()
    dims = list(img_tensor.shape)
    if dims[-2] != dims[-1]:
        click.secho("Warning: Image is not square, a centered square crop will be used instead.",
                    fg = 'yellow', italic = True)
        
    transform = v2.Compose([
        v2.CenterCrop(max(dims[-2:])),
        v2.Resize(32),
        v2.Normalize([127.5, 127.5,127.5], [127.5, 127.5,127.5])
    ])
    
    img_tensor = transform(img_tensor)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, "trained_models\\res_net_18\\res_net_18.pt")
    try:
        if not os.path.exists(model_path):
            merge = Merge(inputdir = 'trained_models/res_net_18',
                          outputdir = 'trained_models/res_net_18',
                          outputfilename = 'res_net_18.pt')
            merge.merge()
            
        net = torch.load(os.path.join(dir_path, "trained_models\\res_net_18\\res_net_18.pt"))
            
    except:
        click.UsageError("Missing res_net_18.pt file, in \'trained_models/res_net_18\'")
    #net = torch.load('D:/Documents/Deep Learning Projects/Cat vs Not Cat Classifier/res_net_18.pth')
    #net = torch.load('res_net_18.pth')
    net = net.to(device)
    net.eval()
    
    output = net(img_tensor[None,:].to(device)).to('cpu')
    is_cat = bool(output.round())
    nl = '\n'
    if is_cat:
        click.echo(f"This is an image of a cat! {nl}Certainty: {round(float(output)*100)}%")
    else:
        click.echo(f"This is an image of a dog! {nl}Certainty: {round((1-float(output))*100)}%")
    
if __name__ == '__main__':
    image_classify()
