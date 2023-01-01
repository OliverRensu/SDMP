import numpy as np
def rand_bbox(img_shape, lam, margin=0., count=None):
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    cy = np.random.randint(0 + cut_h // 2, img_h - cut_h // 2, size=count)
    cx = np.random.randint(0 + cut_w // 2, img_w - cut_w // 2, size=count)
    yl = cy - cut_h // 2
    yh = cy + cut_h // 2
    xl = cx - cut_w // 2
    xh = cx + cut_w // 2
    return yl, yh, xl, xh

yl, yh, xl, xh = rand_bbox([224,224], 0.3)
print((yh - yl) * (xh - xl))
yl, yh, xl, xh = rand_bbox([224,224], 0.3)
print((yh - yl) * (xh - xl))