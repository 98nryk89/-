# Remove white background by flood-filling from the borders (keeps white shirt inside the outline)
from PIL import Image, ImageFilter
import numpy as np
from collections import deque
im = Image.open('assets/hatoriman.png').convert('RGB')
a = np.asarray(im).astype(np.int16)
h, w, _ = a.shape
near_white = (a.min(axis=2) > 232) & ((a.max(axis=2) - a.min(axis=2)) < 18)
bg = np.zeros((h, w), bool)
q = deque()
for x in range(w):
    for y in (0, h-1):
        if near_white[y, x] and not bg[y, x]: bg[y, x] = True; q.append((y, x))
for y in range(h):
    for x in (0, w-1):
        if near_white[y, x] and not bg[y, x]: bg[y, x] = True; q.append((y, x))
for (sx, sy) in [(392, 162), (385, 200), (400, 130)]:  # white pocket enclosed by raised arm and head
    if near_white[sy, sx] and not bg[sy, sx]: bg[sy, sx] = True; q.append((sy, sx))
while q:
    y, x = q.popleft()
    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
        ny, nx = y+dy, x+dx
        if 0 <= ny < h and 0 <= nx < w and not bg[ny, nx] and near_white[ny, nx]:
            bg[ny, nx] = True; q.append((ny, nx))
alpha = Image.fromarray(np.where(bg, 0, 255).astype(np.uint8))
alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.2))
out = im.convert('RGBA'); out.putalpha(alpha)
bbox = out.getbbox(); out = out.crop(bbox)
out.save('assets/hatoriman_cut.png'); print(bbox, out.size)
