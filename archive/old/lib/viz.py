import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw
import math

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def makeRectangle(l, w, theta, offset=(0,0)):
    c, s = math.cos(theta), math.sin(theta)
    rectCoords = [(l/2.0, w/2.0), (l/2.0, -w/2.0), (-l/2.0, -w/2.0), (-l/2.0, w/2.0)]
    return [(c*x-s*y+offset[0], s*x+c*y+offset[1]) for (x,y) in rectCoords]

def plot_results(pil_img, prob, boxes, fills, rotations, id2label):
    #plt.figure(figsize=(16,10))
    #plt.imshow(pil_img)
    #ax = plt.gca()
    #colors = COLORS * 100
    draw = ImageDraw.Draw(pil_img, "RGBA")
    for p, (xmin, ymin, xmax, ymax), (r,g,b), (v1,v2) in zip(prob, boxes.tolist(), fills.tolist(), rotations.tolist()):
        w = xmax - xmin
        h = ymax - ymin
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=np.array([r,g,b]), linewidth=3))
        A  = np.arctan2(v2,v1)
        vertices = makeRectangle(w, h, A, offset=(xmin+w/2, ymin+h/2))  
        draw.polygon(vertices, outline='red')
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f} - {np.array([r,g,b])}'
        draw.text((vertices[2][0], vertices[2][1]), text, fill='black')
#         ax.text(xmin, ymin, text, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.axis('off')
#     plt.show()
    return pil_img


def get_preds(image, outputs, threshold):
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    fills = outputs.pred_fill[0, keep].cpu()
    rotations = outputs.pred_rotation[0, keep].cpu()
    return probas, keep, bboxes_scaled, fills, rotations



def visualize_predictions(image, outputs, id2label, threshold=0.6):
  # keep only predictions with confidence >= threshold
  
    probas, keep, bboxes_scaled, fills, rotations = get_preds(image, outputs, threshold)
    # plot results
    plot_results(image, probas[keep], bboxes_scaled, fills, rotations, id2label)



def img_format(image, feature_extractor):
    image = feature_extractor._resize(image=image, target=None, size=feature_extractor.size, max_size=feature_extractor.max_size)[0]
    image = feature_extractor._normalize(image=image, mean=feature_extractor.image_mean, std=feature_extractor.image_std)[0]
    return torch.tensor(image)