

from .viz import get_preds 
import numpy as np

def check_ag_create(current_slide):
    for shape in current_slide.shapes:
        try:
            if "@AG create" in shape.TextFrame.TextRange.Text:
                shape.Delete()
                return True
        except:
            pass
    return False


def get_readout(image, outputs, id2label, threshold=0.8):
    probas, keep, bboxes_scaled, fills, rotations = get_preds(image, outputs, threshold)
    prob = probas[keep]
    readout = []
    for p, (xmin, ymin, xmax, ymax), (r,g,b), (v1,v2) in zip(prob, bboxes_scaled.tolist(), fills.tolist(), rotations.tolist()):
            x,y, w, h = xmin, ymin, xmax - xmin, ymax - ymin
            cl = p.argmax()
            read = {
                'name': id2label[cl.item()],
                'left': x,
                'top':y,
                'width': w,
                'height':h,
                'fillRGB': [r,g,b],
                'rotation': np.arctan2(v2,v1)
            }
            readout.append(read)
    return readout