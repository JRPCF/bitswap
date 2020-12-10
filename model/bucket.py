import torch
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

def bucket(images, n=1):
    # get Vector Extractor Model
    model = models.vgg16(pretrained=True)
    # remove last layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    
    print(model)
    model.eval()
    
    output = []
    for image in images:
        with torch.no_grad():
            pred=(model( image.float() ))
            pred = torch.flatten(pred)
            output.append(pred.cpu().detach().numpy())
    
    print(". model inference complete")
    
    pca = PCA(n_components=100, random_state=22)
    pca.fit(output)
    x = pca.transform(output)

    print(". PCA complete")
    
    # cluster feature vectors
    kmeans = KMeans(n_clusters=n,n_jobs=-1, random_state=22)
    kmeans.fit(x)
    print(". kmeans complete")
    
    return kmeans