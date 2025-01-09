def bbox_contains(box, x, y):
    return box['x'][0] <= x <= box['x'][1] and box['y'][0] <= y <= box['y'][1]