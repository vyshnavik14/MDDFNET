import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def draw_detection(image_tensor):
    img = (image_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8).copy()
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.imshow(img)
    H, W = img.shape[:2]
    rect = plt.Rectangle((W*0.25, H*0.25), W*0.25, H*0.25, edgecolor='r', facecolor='none', linewidth=3)
    ax.add_patch(rect)
    ax.axis('off')
    plt.show()

def overlay_heatmap(image_tensor, heatmap):
    img = (image_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    hm = heatmap.squeeze()
    hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
    hm_color = cv2.applyColorMap((hm*255).astype('uint8'), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, .6, hm_color, .4, 0)
    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()
