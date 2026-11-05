"""ハトリーマン MAD renderer.

Every frame is drawn procedurally with Pillow from one character cut-out,
synced to the beat grid of assets/song.mp3 (140 BPM, kick on 0.328s + k*0.4286s).
Usage: python3 src/render.py [out.mp4] [--preview t1,t2,...]
"""
import math, random, subprocess, sys, os
from functools import lru_cache
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1080, 1920, 30
DURATION = 17.76
BEAT0, BEAT = 0.328, 0.42857
def B(k): return BEAT0 + k * BEAT

FONT = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
EMOJI = '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'
CHAR = Image.open(os.path.join(ROOT, 'assets/hatoriman_cut.png')).convert('RGBA')
CW, CH = CHAR.size
# landmarks in cut-out coordinates
HEAD = (300, 100); EYE_L = (232, 85); EYE_R = (336, 47); HAND = (160, 105); MASK_GREY = (162, 161, 180)

PINK = (255, 92, 160); HOTPINK = (255, 40, 120); CREAM = (255, 246, 230); NAVY = (30, 32, 70)

# ---------------------------------------------------------------- helpers
def ease_out(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def ease_in_out(x): x = min(max(x, 0), 1); return 3 * x * x - 2 * x * x * x
def lerp(a, b, x): return a + (b - a) * x
def pop(x, over=1.25):
    """0->overshoot->1 scale curve"""
    x = min(max(x, 0), 1)
    if x < 0.6: return ease_out(x / 0.6) * over
    return over + (1 - over) * ease_in_out((x - 0.6) / 0.4)

@lru_cache(64)
def font(size): return ImageFont.truetype(FONT, size)

def new(c=(0, 0, 0, 0)): return Image.new('RGBA', (W, H), c)

def paste(dst, src, x, y, alpha=1.0):
    """alpha-composite src with its top-left at (x,y), clipping safely."""
    x, y = int(round(x)), int(round(y))
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    w = min(src.width - sx0, dst.width - dx0); h = min(src.height - sy0, dst.height - dy0)
    if w <= 0 or h <= 0: return
    piece = src.crop((sx0, sy0, sx0 + w, sy0 + h))
    if alpha < 1:
        a = piece.getchannel('A').point(lambda v: int(v * alpha)); piece = piece.copy(); piece.putalpha(a)
    dst.alpha_composite(piece, (dx0, dy0))

def paste_center(dst, src, cx, cy, alpha=1.0): paste(dst, src, cx - src.width / 2, cy - src.height / 2, alpha)

def vgrad(c1, c2, h=H, w=W):
    t = np.linspace(0, 1, h)[:, None, None]
    a = (np.array(c1)[None, None] * (1 - t) + np.array(c2)[None, None] * t).repeat(w, 1)
    return Image.fromarray(a.astype(np.uint8), 'RGB').convert('RGBA')

def radial(c_in, c_out, cx=W / 2, cy=H / 2, r=1100, power=1.0):
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.clip(np.hypot(xx - cx, yy - cy) / r, 0, 1)[..., None] ** power
    a = np.array(c_in)[None, None] * (1 - d) + np.array(c_out)[None, None] * d
    return Image.fromarray(a.astype(np.uint8), 'RGB').convert('RGBA')

@lru_cache(8)
def vignette(color=(255, 70, 150), strength=200, r=1150):
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.clip((np.hypot((xx - W / 2) * 1.5, yy - H / 2) - r * 0.55) / (r * 0.6), 0, 1) ** 1.6
    a = np.zeros((H, W, 4), np.uint8); a[..., :3] = color; a[..., 3] = (d * strength).astype(np.uint8)
    return Image.fromarray(a, 'RGBA')

# ---------------------------------------------------------------- character
@lru_cache(256)
def char_img(scale, angle=0.0, flip=False, crop=None, mode='normal', smile=False, blush=False):
    im = CHAR
    if smile or blush: im = face_mod(smile, blush)
    if crop:
        im = im.crop(crop)
        fe = np.ones((im.height, im.width), np.float32)
        for side, inside in (('l', crop[0] > 0), ('t', crop[1] > 0), ('r', crop[2] < CW), ('b', crop[3] < CH)):
            if not inside: continue
            n = int((im.width if side in 'lr' else im.height) * (0.16 if side in 'lr' else 0.22))
            ramp = np.linspace(0, 1, n) ** 1.5
            if side == 'l': fe[:, :n] *= ramp[None]
            if side == 'r': fe[:, -n:] *= ramp[::-1][None]
            if side == 't': fe[:n] *= ramp[:, None]
            if side == 'b': fe[-n:] *= ramp[::-1][:, None]
        a = (np.asarray(im.getchannel('A')).astype(np.float32) * fe).astype(np.uint8)
        im = im.copy(); im.putalpha(Image.fromarray(a))
    if flip: im = ImageOps.mirror(im)
    if mode == 'silhouette':
        a = im.getchannel('A'); im = Image.new('RGBA', im.size, (8, 8, 16, 255)); im.putalpha(a)
    elif mode == 'red':
        g = ImageOps.grayscale(im.convert('RGB')); r = Image.merge('RGB', (g.point(lambda v: min(255, v + 60)), g.point(lambda v: v // 3), g.point(lambda v: v // 3)))
        a = im.getchannel('A'); im = r.convert('RGBA'); im.putalpha(a)
    elif mode == 'mono':
        g = ImageOps.grayscale(im.convert('RGB')).point(lambda v: 255 if v > 110 else 0)
        a = im.getchannel('A'); im = g.convert('RGBA'); im.putalpha(a)
    elif mode == 'invert':
        rgb = ImageOps.invert(im.convert('RGB')); a = im.getchannel('A'); im = rgb.convert('RGBA'); im.putalpha(a)
    elif mode == 'warm':
        rgb = im.convert('RGB'); r, g, b = rgb.split()
        rgb = Image.merge('RGB', (r.point(lambda v: min(255, int(v * 1.12 + 12))), g.point(lambda v: min(255, int(v * 1.02 + 4))), b.point(lambda v: int(v * 0.86))))
        a = im.getchannel('A'); im = rgb.convert('RGBA'); im.putalpha(a)
    elif mode == 'dark':
        rgb = im.convert('RGB').point(lambda v: int(v * 0.55)); a = im.getchannel('A'); im = rgb.convert('RGBA'); im.putalpha(a)
    w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im = im.resize((w, h), Image.LANCZOS if scale < 1 else Image.BICUBIC)
    if angle: im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    return im

@lru_cache(4)
def face_mod(smile, blush):
    im = CHAR.copy(); d = ImageDraw.Draw(im)
    if blush:
        for (x, y) in (EYE_L, EYE_R):
            bl = Image.new('RGBA', im.size); bd = ImageDraw.Draw(bl)
            bd.ellipse((x - 26, y + 22, x + 22, y + 44), fill=(255, 90, 130, 150))
            for i in range(3): bd.line((x - 14 + i * 12, y + 40, x - 6 + i * 12, y + 26), fill=(230, 40, 90, 230), width=3)
            im.alpha_composite(bl.filter(ImageFilter.GaussianBlur(1.2)))
    if smile:
        for (x, y) in (EYE_L, EYE_R):
            d.ellipse((x - 19, y - 19, x + 19, y + 19), fill=MASK_GREY + (255,))
            d.arc((x - 15, y - 8, x + 15, y + 20), 200, 340, fill=(30, 25, 30, 255), width=6)
    return im

def put_char(dst, cx, cy, scale, angle=0.0, flip=False, crop=None, mode='normal', alpha=1.0, smile=False, blush=False, anchor='center'):
    s = round(scale, 3); a = round(angle * 2) / 2
    im = char_img(s, a, flip, crop, mode, smile, blush)
    if anchor == 'center': paste_center(dst, im, cx, cy, alpha)
    else: paste(dst, im, cx - im.width / 2, cy - im.height, alpha)  # bottom-centre
    return im

def ghost_trail(dst, cx, cy, scale, angle, dx, dy, n=4, flip=False, mode='normal'):
    for i in range(n, 0, -1):
        put_char(dst, cx - dx * i, cy - dy * i, scale, angle, flip, mode=mode, alpha=0.16 * (n - i + 1) / n)

# ---------------------------------------------------------------- graphic fx
@lru_cache(32)
def focus_lines(seed, color=(0, 0, 0), n=150, inner=330, cx=W // 2, cy=H // 2, alpha=255, ry=1.25):
    rnd = random.Random(seed); im = new(); d = ImageDraw.Draw(im)
    R = 2400
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi); wdt = rnd.uniform(0.004, 0.018); r0 = inner * rnd.uniform(0.9, 1.6)
        p0 = (cx + math.cos(a) * r0, cy + math.sin(a) * r0 * ry)
        p1 = (cx + math.cos(a - wdt) * R, cy + math.sin(a - wdt) * R)
        p2 = (cx + math.cos(a + wdt) * R, cy + math.sin(a + wdt) * R)
        d.polygon([p0, p1, p2], fill=color + (alpha,))
    return im

@lru_cache(16)
def speed_lines(seed, angle_deg, color=(255, 255, 255), n=70, alpha=200):
    rnd = random.Random(seed); big = Image.new('RGBA', (2600, 2600)); d = ImageDraw.Draw(big)
    for _ in range(n):
        y = rnd.uniform(0, 2600); x = rnd.uniform(-400, 2200); L = rnd.uniform(500, 1600); w = rnd.uniform(2, 9)
        d.polygon([(x, y), (x + L, y - w), (x + L, y + w)], fill=color + (alpha,))
    big = big.rotate(angle_deg, resample=Image.BICUBIC)
    return big.crop((760, 340, 760 + W, 340 + H))

def heart_poly(cx, cy, s, rot=0.0):
    pts = []
    for i in range(48):
        t = i / 48 * 2 * math.pi
        x = 16 * math.sin(t) ** 3; y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        xr = x * math.cos(rot) - y * math.sin(rot); yr = x * math.sin(rot) + y * math.cos(rot)
        pts.append((cx + xr * s / 16, cy + yr * s / 16))
    return pts

def heart(d, cx, cy, s, fill=HOTPINK, outline=(255, 255, 255), rot=0.0, width=0):
    d.polygon(heart_poly(cx, cy, s, rot), fill=fill, outline=outline if width else None, width=width)

def sparkle(d, cx, cy, s, fill=(255, 255, 255, 255)):
    k = s * 0.22
    d.polygon([(cx, cy - s), (cx + k, cy - k), (cx + s, cy), (cx + k, cy + k), (cx, cy + s), (cx - k, cy + k), (cx - s, cy), (cx - k, cy - k)], fill=fill)

def heart_burst(dst, t, seed, n=24, cx=W / 2, cy=H / 2, spread=900, colors=None, size=(30, 80)):
    rnd = random.Random(seed); lay = new(); d = ImageDraw.Draw(lay)
    colors = colors or [HOTPINK, (255, 140, 190), (255, 255, 255), (255, 60, 90)]
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi); sp = rnd.uniform(0.4, 1.0) * spread; s = rnd.uniform(*size)
        k = ease_out(t * rnd.uniform(0.8, 1.3))
        x = cx + math.cos(a) * sp * k; y = cy + math.sin(a) * sp * k - 120 * t
        c = rnd.choice(colors)
        heart(d, x, y, s * (1 - 0.4 * t), fill=c + (int(255 * (1 - t * 0.6)),), rot=rnd.uniform(-0.5, 0.5))
    dst.alpha_composite(lay)

def floating_hearts(dst, t, seed, n=14, alpha=200):
    rnd = random.Random(seed); lay = new(); d = ImageDraw.Draw(lay)
    for i in range(n):
        x = rnd.uniform(40, W - 40); y0 = rnd.uniform(0, H); sp = rnd.uniform(150, 420); s = rnd.uniform(18, 55)
        y = (y0 - sp * t) % (H + 200) - 100; x += math.sin(t * 3 + i) * 30
        heart(d, x, y, s, fill=rnd.choice([HOTPINK, (255, 150, 200), (255, 255, 255)]) + (alpha,))
    dst.alpha_composite(lay)

def sparkles(dst, t, seed, n=18, area=(0, 0, W, H), color=(255, 255, 255)):
    rnd = random.Random(seed); lay = new(); d = ImageDraw.Draw(lay)
    for _ in range(n):
        x = rnd.uniform(area[0], area[2]); y = rnd.uniform(area[1], area[3]); s = rnd.uniform(10, 38); ph = rnd.uniform(0, 6.28)
        k = 0.5 + 0.5 * math.sin(t * 14 + ph)
        sparkle(d, x, y, s * (0.3 + 0.7 * k), fill=color + (int(255 * k),))
    dst.alpha_composite(lay)

def halftone(color=(0, 0, 0), step=22, alpha=60, seed=0):
    return _halftone(color, step, alpha)

@lru_cache(8)
def _halftone(color, step, alpha):
    im = new(); d = ImageDraw.Draw(im)
    for y in range(0, H, step):
        for x in range((y // step % 2) * step // 2, W, step):
            r = step * 0.28 * (0.4 + 0.6 * y / H)
            d.ellipse((x - r, y - r, x + r, y + r), fill=color + (alpha,))
    return im

# ---------------------------------------------------------------- text
def text(dst, s, cx, cy, size, fill=(255, 255, 255), stroke=(0, 0, 0), sw=10, angle=0.0, scale=1.0, alpha=1.0, bold=2, spacing=8):
    size = max(8, int(size * scale)); f = font(size)
    tmp = Image.new('RGBA', (10, 10)); d = ImageDraw.Draw(tmp)
    bb = d.multiline_textbbox((0, 0), s, font=f, stroke_width=sw + bold, spacing=spacing, align='center')
    w, h = bb[2] - bb[0] + 20, bb[3] - bb[1] + 20
    im = Image.new('RGBA', (w, h)); d = ImageDraw.Draw(im)
    o = (10 - bb[0], 10 - bb[1])
    if sw: d.multiline_text(o, s, font=f, fill=stroke, stroke_width=sw + bold, stroke_fill=stroke, spacing=spacing, align='center')
    d.multiline_text(o, s, font=f, fill=fill, stroke_width=bold, stroke_fill=fill, spacing=spacing, align='center')
    if angle: im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    paste_center(dst, im, cx, cy, alpha)
    return im.size

def sfx(dst, s, cx, cy, size, t, fill=(255, 255, 255), stroke=(0, 0, 0), angle=-8, shake=True):
    """Manga sound effect that slams in."""
    sc = pop(t / 0.35, 1.45)
    j = (math.sin(t * 90) * 6) if shake and t < 0.3 else 0
    text(dst, s, cx + j, cy, size, fill=fill, stroke=stroke, sw=14, angle=angle, scale=sc, bold=4)

def bubble(dst, s, cx, cy, size, t, tail=(0, 120), jagged=False, fill=(255, 255, 255), ink=(20, 20, 30)):
    k = pop(t / 0.22, 1.15)
    if k <= 0.01: return
    f = font(int(size)); d0 = ImageDraw.Draw(Image.new('RGBA', (4, 4)))
    bb = d0.multiline_textbbox((0, 0), s, font=f, spacing=10, align='center')
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    rw, rh = tw * 0.72 + 50, th * 0.75 + 45
    lay = new(); d = ImageDraw.Draw(lay)
    if jagged:
        pts = []; rnd = random.Random(len(s))
        for i in range(28):
            a = i / 28 * 2 * math.pi; r = 1.0 if i % 2 == 0 else 0.78 + rnd.uniform(-0.05, 0.05)
            pts.append((cx + math.cos(a) * rw * 1.18 * r, cy + math.sin(a) * rh * 1.25 * r))
        d.polygon(pts, fill=fill + (255,), outline=ink + (255,), width=6)
    else:
        tx, ty = cx + tail[0], cy + tail[1]
        d.polygon([(cx - 40, cy + rh * 0.6), (cx + 30, cy + rh * 0.6), (tx, ty)], fill=fill + (255,), outline=ink + (255,), width=6)
        d.ellipse((cx - rw, cy - rh, cx + rw, cy + rh), fill=fill + (255,), outline=ink + (255,), width=6)
        d.polygon([(cx - 34, cy + rh * 0.55), (cx + 24, cy + rh * 0.55), (tx * 0.15 + cx * 0.85, ty * 0.15 + cy * 0.85 + 10)], fill=fill + (255,))
    d.multiline_text((cx - tw / 2 - bb[0], cy - th / 2 - bb[1]), s, font=f, fill=ink + (255,), spacing=10, align='center', stroke_width=1, stroke_fill=ink + (255,))
    if k != 1:
        lay = lay.resize((int(W * k), int(H * k)), Image.BILINEAR)
        tmp = new(); paste(tmp, lay, cx - cx * k, cy - cy * k); lay = tmp
    dst.alpha_composite(lay)

def monologue(dst, s, t, y=1380, size=54):
    a = min(1, t / 0.12)
    band = Image.new('RGBA', (W, int(size * 2.1)), (20, 0, 20, int(110 * a)))
    paste(dst, band, 0, y - band.height / 2)
    text(dst, s, W / 2, y, size, fill=(255, 235, 245), stroke=(120, 20, 70), sw=5, alpha=a, bold=1)

LYRICS = [  # (start, end-of-typing, text) — sung line telop; each holds until the next starts
    (0.33, 1.95, '幸せなのでも'),
    (B(4), B(8), 'chu chu chu chu'),
    (B(8), B(17), '知らない感情で埋もれそう'),
    (B(20), B(24), 'chu chu chu chu'),
    (B(24), B(33), 'こんなに臆病だっけ？'),
]
def lyric_telop(dst, t):
    cur = None
    for i, (s, e, txt) in enumerate(LYRICS):
        if t >= s: cur = (s, e, txt)
    if cur is None: return
    s, e, txt = cur
    if txt.startswith('chu'):
        n = min(4, int((t - s) / BEAT) + 1); shown = ' '.join(['chu'] * n)
    else:
        n = max(1, min(len(txt), int(math.ceil((t - s) / (e - s) * len(txt))))); shown = txt[:n]
    size = 84 if len(txt) < 10 else 72
    text(dst, shown, W / 2, 1700, size, fill=(255, 255, 255), stroke=(235, 40, 120), sw=9, bold=2)

@lru_cache(4)
def emoji_img(ch, size):
    f = ImageFont.truetype(EMOJI, 109); im = Image.new('RGBA', (140, 140)); ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox()); return im.resize((size, int(size * im.height / im.width)), Image.LANCZOS)

def emoji(dst, ch, cx, cy, size, angle=0, alpha=1.0):
    im = emoji_img(ch, int(size))
    if angle: im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    paste_center(dst, im, cx, cy, alpha)

# ---------------------------------------------------------------- props
def thug(d, x, y, s=1.0, col=(10, 6, 14), eye=(255, 40, 40), lean=0.0, bat=True):
    """Delinquent silhouette (pompadour + long coat). (x,y)=feet centre."""
    def P(px, py): return (x + (px - lean * py) * s, y + py * s)
    d.polygon([P(-62, 0), P(-28, 0), P(-12, -400), P(-58, -400)], fill=col)                 # leg L
    d.polygon([P(28, 0), P(66, 0), P(58, -400), P(12, -400)], fill=col)                     # leg R
    d.polygon([P(-52, 6), P(-100, 6), P(-100, -14), P(-40, -14)], fill=col); d.polygon([P(52, 6), P(104, 6), P(104, -14), P(40, -14)], fill=col)
    d.polygon([P(-112, -300), P(-70, -460), P(-104, -655), P(104, -655), P(70, -460), P(112, -300)], fill=col)  # flared long coat
    d.polygon([P(-104, -650), P(-150, -560), P(-142, -400), P(-112, -400), P(-112, -540), P(-80, -600)], fill=col)  # arm L
    d.polygon([P(104, -650), P(150, -560), P(142, -400), P(112, -400), P(112, -540), P(80, -600)], fill=col)    # arm R
    d.polygon([P(-30, -660), P(30, -660), P(24, -690), P(-24, -690)], fill=col)             # neck
    d.ellipse((*P(-50, -790), *P(50, -680)), fill=col)                                       # head
    d.ellipse((*P(-46, -838), *P(118, -768)), fill=col)                                      # pompadour
    d.line((*P(-28, -742), *P(-6, -734)), fill=eye, width=max(2, int(8 * s)))
    d.line((*P(10, -734), *P(32, -742)), fill=eye, width=max(2, int(8 * s)))
    if bat: d.line((*P(130, -410), *P(230, -800)), fill=(150, 150, 160), width=max(3, int(18 * s)))

def rope(d, y, t=0.0, cut=False):
    col = (176, 132, 80); dk = (110, 76, 40)
    for i, (x0, x1, dy) in enumerate([(-50, W + 50, 0), (-50, W + 50, 70)]):
        if cut:
            fall = ease_out(t) * 500
            d.line((x0, y + dy + fall, W / 2 - 40, y + dy + fall * 1.3), fill=col, width=30)
            d.line((W / 2 + 40, y + dy + fall * 1.2, x1, y + dy + fall), fill=col, width=30)
        else:
            d.line((x0, y + dy - 40, x1, y + dy + 40), fill=col, width=30)
            for k in range(0, W, 40): d.line((k, y + dy - 52 + k * 80 / W, k + 20, y + dy - 28 + k * 80 / W), fill=dk, width=4)

# ---------------------------------------------------------------- backgrounds
@lru_cache(1)
def bg_morning():
    im = vgrad((150, 200, 255), (255, 220, 200)); d = ImageDraw.Draw(im); rnd = random.Random(3)
    sun = radial((255, 255, 230), (255, 220, 200), 820, 420, 600, 0.7).getchannel('R')
    for x in range(-50, W, 150):
        h = rnd.randint(500, 1050); c = rnd.choice([(120, 140, 190), (140, 160, 205), (105, 120, 170)])
        d.rectangle((x, 1350 - h, x + rnd.randint(120, 170), 1450), fill=c)
        for wy in range(1350 - h + 30, 1420, 55):
            for wx in range(x + 18, x + 120, 38): d.rectangle((wx, wy, wx + 18, wy + 28), fill=(220, 235, 255))
    d.rectangle((0, 1450, W, H), fill=(205, 205, 215))
    for k in range(9): d.line((0, 1480 + k * 60, W, 1480 + k * 60), fill=(185, 185, 198), width=4)
    d.rectangle((0, 1440, W, 1470), fill=(240, 200, 40))                        # platform tactile line
    d.rectangle((60, 260, 520, 380), fill=(30, 90, 60)); d.rectangle((70, 270, 510, 370), outline=(255, 255, 255), width=4)
    text(im, 'はと町駅', 290, 320, 64, fill=(255, 255, 255), stroke=(30, 90, 60), sw=0, bold=1)
    return im

@lru_cache(1)
def bg_office():
    im = vgrad((210, 222, 240), (170, 185, 210)); d = ImageDraw.Draw(im)
    for y in range(120, 1000, 46): d.rectangle((80, y, 1000, y + 30), fill=(232, 240, 252))      # blinds
    d.rectangle((60, 100, 1020, 1020), outline=(140, 150, 175), width=14)
    d.rectangle((0, 1250, W, H), fill=(120, 95, 80)); d.rectangle((0, 1250, W, 1285), fill=(150, 120, 100))
    return im

@lru_cache(1)
def bg_rain_base(): return vgrad((40, 60, 100), (90, 110, 150))

def rain(dst, t, seed=5, n=160, col=(200, 220, 255, 150)):
    rnd = random.Random(seed); d = ImageDraw.Draw(dst)
    for _ in range(n):
        x = rnd.uniform(-200, W); y = (rnd.uniform(0, H) + t * 2600) % (H + 200) - 100; L = rnd.uniform(40, 110)
        d.line((x, y, x + L * 0.25, y + L), fill=col, width=3)

@lru_cache(4)
def bg_bokeh(c1, c2, seed):
    im = radial(c1, c2, W / 2, H * 0.45, 1300); rnd = random.Random(seed); lay = new(); d = ImageDraw.Draw(lay)
    for _ in range(26):
        x, y, r = rnd.uniform(0, W), rnd.uniform(0, H), rnd.uniform(40, 150)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, rnd.randint(25, 70)))
    im.alpha_composite(lay.filter(ImageFilter.GaussianBlur(6))); return im

@lru_cache(1)
def bg_alley():
    im = vgrad((30, 10, 40), (70, 20, 50)); d = ImageDraw.Draw(im)
    for y in range(0, 1500, 70):
        for x in range((y // 70 % 2) * 60, W, 120): d.rectangle((x, y, x + 112, y + 62), outline=(50, 22, 55), width=4)
    d.rectangle((0, 1500, W, H), fill=(25, 12, 28))
    lamp = radial((255, 200, 120), (0, 0, 0), 820, 260, 700, 1.4).getchannel('R')
    glow = Image.new('RGBA', (W, H), (255, 170, 90, 0)); glow.putalpha(lamp.point(lambda v: int(v * 0.55)))
    im.alpha_composite(glow); d.rectangle((800, 240, 840, 1500), fill=(20, 10, 22))
    return im

@lru_cache(1)
def bg_warehouse():
    im = vgrad((14, 14, 22), (34, 30, 40)); d = ImageDraw.Draw(im); rnd = random.Random(9)
    for i in range(7):
        x = rnd.randint(-100, W - 150); y = rnd.randint(900, 1350); s = rnd.randint(200, 330)
        d.rectangle((x, y, x + s, y + s), fill=(46, 38, 34), outline=(24, 20, 18), width=8)
        d.line((x, y, x + s, y + s), fill=(30, 25, 22), width=8)
    for x in range(0, W, 180): d.line((x, 0, x - 120, 760), fill=(28, 28, 38), width=24)  # beams
    return im

def door(dst, open_k, light=0.0, crack=0.0):
    """Warehouse door at top-centre; returns light polygon layer."""
    d = ImageDraw.Draw(dst); x0, y0, x1, y1 = 300, 380, 780, 1180
    if open_k <= 0:
        d.rectangle((x0, y0, x1, y1), fill=(60, 58, 66), outline=(20, 20, 26), width=12)
        for y in range(y0 + 40, y1, 60): d.line((x0 + 10, y, x1 - 10, y), fill=(45, 44, 52), width=6)
        if crack > 0:
            g = new(); gd = ImageDraw.Draw(g)
            gd.rectangle((x0 - 6, y0, x0 + 6, y1), fill=(255, 245, 210, int(200 * crack)))
            gd.rectangle((x0, y1 - 8, x1, y1 + 8), fill=(255, 245, 210, int(220 * crack)))
            dst.alpha_composite(g.filter(ImageFilter.GaussianBlur(10)))
    else:
        g = new(); gd = ImageDraw.Draw(g)
        gd.rectangle((x0, y0, x1, y1), fill=(255, 250, 225, 255))
        gd.polygon([(x0, y1), (x1, y1), (W + 300, H), (-300, H)], fill=(255, 245, 200, int(110 * light)))
        dst.alpha_composite(g.filter(ImageFilter.GaussianBlur(4)))
        dst.alpha_composite(focus_lines(77, (255, 250, 220), 60, 260, W // 2, 780, 90, 1.0))

@lru_cache(1)
def bg_sunset():
    im = vgrad((255, 140, 120), (255, 214, 150)); d = ImageDraw.Draw(im); rnd = random.Random(11)
    sun = radial((255, 250, 220), (0, 0, 0), W / 2, 1180, 420, 1.2).getchannel('R')
    g = Image.new('RGBA', (W, H), (255, 245, 210, 0)); g.putalpha(sun); im.alpha_composite(g)
    for x in range(-40, W, 110):
        h = rnd.randint(150, 520); d.rectangle((x, 1400 - h, x + rnd.randint(80, 120), 1500), fill=(120, 60, 90))
    d.rectangle((0, 1490, W, H), fill=(90, 40, 70))
    cl = new(); cd = ImageDraw.Draw(cl)
    for _ in range(9):
        x, y = rnd.uniform(-100, W), rnd.uniform(150, 800); cd.ellipse((x, y, x + rnd.uniform(200, 420), y + rnd.uniform(40, 80)), fill=(255, 220, 220, 140))
    im.alpha_composite(cl.filter(ImageFilter.GaussianBlur(8))); return im

def phone_frame(dst, inner, t, likes):
    """Generic SNS phone UI with `inner` (W*0.78 x H*0.7) as video content."""
    px, py, pw, ph = 110, 200, 860, 1520
    lay = new(); d = ImageDraw.Draw(lay)
    d.rounded_rectangle((px - 22, py - 22, px + pw + 22, py + ph + 22), 80, fill=(18, 18, 22, 255))
    mask = Image.new('L', (pw, ph)); ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw, ph), 60, fill=255)
    scr = inner.resize((pw, ph)); lay.paste(scr, (px, py), mask)
    d.rounded_rectangle((px + pw / 2 - 90, py + 18, px + pw / 2 + 90, py + 52), 18, fill=(18, 18, 22, 255))
    d.ellipse((px + 30, py + ph - 260, px + 110, py + ph - 180), fill=(170, 170, 190, 255))
    text(lay, '@hatoriman_official', px + 330, py + ph - 220, 40, sw=4, bold=1)
    text(lay, '#踊ってみた #ハトリーマン #定時ダンス', px + 400, py + ph - 140, 34, sw=4, bold=1)
    heart(d, px + pw - 80, py + ph - 520, 46, fill=(255, 50, 90, 255))
    text(lay, likes, px + pw - 80, py + ph - 445, 34, sw=4, bold=1)
    d.ellipse((px + pw - 112, py + ph - 380, px + pw - 48, py + ph - 330), fill=(255, 255, 255, 230))
    text(lay, '● LIVE', px + 110, py + 110, 34, fill=(255, 255, 255), stroke=(230, 30, 60), sw=8, bold=1)
    dst.alpha_composite(lay)

# ---------------------------------------------------------------- scenes
# each scene: f(t_local, dur, t_global) -> RGBA frame
def sc_commute(lt, dur, t):
    im = bg_morning().copy()
    k = ease_out(lt / (dur * 0.9)); x = lerp(1350, 690, k); bob = abs(math.sin(lt * 7.3)) * -18
    put_char(im, x, 2120 + bob, 1.2, angle=math.sin(lt * 7.3) * 1.5, anchor='bottom')
    im.alpha_composite(vignette((255, 120, 170), 120))
    text(im, 'AM 8:12', 900, 140, 46, fill=(255, 255, 255), stroke=(60, 80, 140), sw=6, bold=1)
    monologue(im, '（あ…今日もとなりに来てくれた）', lt)
    sparkles(im, t, 4, 10, (300, 200, 1080, 1100))
    return im

def sc_thumbs(lt, dur, t):
    im = bg_bokeh((255, 240, 220), (255, 170, 200), 1)
    z = lerp(1.0, 1.08, lt / dur)
    put_char(im, 590, 1040, 1.95 * z, crop=(0, 0, CW, 760))
    # thumbs-up over the raised hand
    hx = 590 + (HAND[0] - CW / 2) * 1.95 * z; hy = 1040 + (HAND[1] - 380) * 1.95 * z
    k = pop((lt - 0.05) / 0.3, 1.35)
    if k > 0:
        lay = new(); d = ImageDraw.Draw(lay); d.ellipse((hx - 170 * k, hy - 170 * k, hx + 170 * k, hy + 170 * k), fill=(255, 250, 200, 140)); im.alpha_composite(lay.filter(ImageFilter.GaussianBlur(20)))
        emoji(im, '👍', hx, hy, 260 * k, angle=12)
        sfx(im, 'グッ', hx + 170, hy - 230, 120, lt, fill=(255, 220, 60), stroke=(120, 50, 0))
    bubble(im, 'おはよ。', 830, 380, 70, lt - 0.15, tail=(-120, 160))
    sparkles(im, t, 7, 22)
    floating_hearts(im, lt, 2, 10, 170)
    im.alpha_composite(vignette((255, 90, 160), 150))
    return im

CHU_COLORS = [((255, 120, 180), (255, 210, 235)), ((255, 220, 90), (255, 250, 210)), ((120, 230, 210), (230, 255, 248)), ((190, 150, 255), (240, 230, 255))]
def make_chu(i):
    def f(lt, dur, t):
        c1, c2 = CHU_COLORS[i]; im = radial(c2, c1, W / 2, H * 0.42, 1200)
        im.alpha_composite(focus_lines(100 + i, (255, 255, 255), 120, 380, alpha=200))
        z = 1.0 + 0.1 * ease_out(lt / dur)
        ang = [8, -10, 14, -6][i]; flip = i % 2 == 1
        put_char(im, W / 2 + [-30, 40, 0, -20][i], 760, [2.6, 3.0, 2.4, 3.4][i] * z, angle=ang, flip=flip, crop=(150, 0, 470, 330), blush=True)
        heart_burst(im, min(1, lt / dur * 1.2), 200 + i, 18, W / 2, 760, 700)
        text(im, 'chu♡', W / 2 + [-180, 190, -160, 170][i], [1280, 1250, 1300, 1260][i], 150, fill=(255, 255, 255), stroke=HOTPINK, sw=14, angle=[10, -12, 8, -6][i], scale=pop(lt / 0.2, 1.3), bold=4)
        return im
    return f

def sc_office(lt, dur, t):
    im = bg_office().copy(); d = ImageDraw.Draw(im)
    put_char(im, 560, 820, 1.55, crop=(0, 0, CW, 760), angle=-3)
    d.rectangle((0, 1180, W, 1290), fill=(150, 120, 100)); d.rectangle((0, 1290, W, H), fill=(110, 85, 70))
    d.polygon([(620, 1190), (1000, 1190), (980, 900), (640, 900)], fill=(40, 44, 60)); d.rectangle((650, 910, 970, 1170), fill=(120, 190, 255))
    for k in range(6): d.line((680, 950 + k * 35, 680 + (k * 97 % 250), 950 + k * 35), fill=(255, 255, 255), width=8)
    d.rectangle((150, 1150, 330, 1185), fill=(230, 230, 230))
    text(im, '22:47', 160, 180, 60, fill=(255, 255, 255), stroke=(60, 70, 110), sw=6, bold=1)
    monologue(im, '（残業中の横顔、ずるい）', lt)
    sparkles(im, t, 12, 8, (200, 200, 800, 900))
    return im

def sc_umbrella(lt, dur, t):
    im = bg_rain_base().copy(); rain(im, t)
    put_char(im, 540, 2060, 1.25, flip=True, anchor='bottom', mode='normal')
    lay = new(); d = ImageDraw.Draw(lay)
    d.pieslice((40, 230, 1040, 900), 180, 360, fill=(255, 110, 160, 255)); d.line((540, 560, 540, 1250), fill=(60, 40, 40, 255), width=14)
    for k in range(1, 6): d.line((540, 230 + 6, 40 + k * 1000 / 6, 565), fill=(220, 70, 120, 255), width=5)
    im.alpha_composite(lay); rain(im, t, 6, 60, (230, 240, 255, 200))
    bubble(im, '入りなよ', 260, 1050, 66, lt - 0.05, tail=(120, 140))
    im.alpha_composite(vignette((90, 120, 200), 170))
    return im

def sc_comfort(lt, dur, t):
    im = bg_bokeh((250, 230, 210), (170, 150, 200), 3)
    z = lerp(1.0, 1.12, ease_in_out(lt / dur))
    put_char(im, 520, 1150, 2.3 * z, angle=-12, crop=(0, 0, CW, 700), mode='warm')
    lay = new(); d = ImageDraw.Draw(lay)  # my tears at the bottom of the view
    for i, x in enumerate([180, 900]):
        yy = 1500 + (lt * 500 + i * 150) % 300
        d.ellipse((x - 24, yy - 30, x + 24, yy + 30), fill=(170, 220, 255, 180)); d.polygon([(x - 20, yy - 15), (x + 20, yy - 15), (x, yy - 70)], fill=(170, 220, 255, 180))
    im.alpha_composite(lay)
    bubble(im, '大丈夫。\nちゃんと見てるよ', 700, 330, 62, lt - 0.05, tail=(-150, 190))
    if lt > dur * 0.45: monologue(im, '（…ずっと、優しい）', lt - dur * 0.45, y=1450)
    sparkles(im, t, 21, 16, color=(255, 250, 220))
    im.alpha_composite(vignette((255, 150, 190), 140))
    return im

def sc_yankee(lt, dur, t):
    im = bg_alley().copy(); lay = new(); d = ImageDraw.Draw(lay)
    for i, (x, s) in enumerate([(250, 1.35), (560, 1.6), (870, 1.35)]):
        thug(d, x + math.sin(lt * 9 + i) * 8, 1880 + 30 * (i != 1), s, lean=0.03 * (i - 1), bat=i != 1)
    im.alpha_composite(lay)
    im.alpha_composite(vignette((200, 0, 30), 200))
    bubble(im, 'よぉ、ねーちゃん\nちょっと来いよ', W / 2, 330, 62, lt, jagged=True)
    return im

def sc_flykick(lt, dur, t, flip=False, sfx_txt='ドゴッ!!', color=(255, 60, 60)):
    im = radial((255, 240, 200), color, W / 2, H / 2, 1200)
    im.alpha_composite(speed_lines(31 + flip, -35 if not flip else 35, (255, 255, 255), 80, 230))
    k = ease_out(lt / (dur * 0.5)); sgn = -1 if flip else 1
    x = lerp(W / 2 - sgn * 900, W / 2, k); y = lerp(1500, 860, k)
    ang = 118 if not flip else -118
    ghost_trail(im, x, y, 0.78, ang, -sgn * 120, 70, 4, flip)
    put_char(im, x, y, 0.78, ang, flip)
    if lt > dur * 0.35: sfx(im, sfx_txt, W / 2, 1450, 190, lt - dur * 0.35, fill=(255, 255, 255), stroke=(20, 0, 0))
    return im

def sc_impact(lt, dur, t, txt='バキィ!!', who='thugs'):
    neg = int(lt * FPS) % 4 < 2
    im = new((255, 255, 255, 255) if neg else (0, 0, 0, 255))
    im.alpha_composite(focus_lines(55, (0, 0, 0) if neg else (255, 255, 255), 170, 300))
    lay = new(); d = ImageDraw.Draw(lay); col = (0, 0, 0) if neg else (255, 255, 255)
    for i, (dx, rot) in enumerate([(-260, -30), (240, 25)]):
        tl = new(); td = ImageDraw.Draw(tl); thug(td, W / 2, 1500, 0.9, col=col, eye=(255, 0, 0), bat=False)
        tl = tl.rotate(rot + lt * 200 * (1 if dx > 0 else -1), center=(W / 2, 1100)); paste(lay, tl, dx * (1 + lt * 3), -lt * 600)
    im.alpha_composite(lay)
    put_char(im, W / 2 - 60 + lt * 200, 1150, 0.7, 112, mode='silhouette' if neg else 'mono')
    sfx(im, txt, W / 2, 520, 220, lt, fill=(255, 40, 40), stroke=(255, 255, 255) if not neg else (0, 0, 0), angle=10)
    return im

def sc_guard(lt, dur, t):
    im = bg_alley().copy(); im.alpha_composite(focus_lines(61, (120, 180, 255), 90, 500, alpha=110))
    z = lerp(1.0, 1.06, lt / dur)
    put_char(im, W / 2, 1990, 1.35 * z, flip=True, anchor='bottom')
    sparkles(im, t, 13, 10, color=(180, 220, 255))
    bubble(im, '下がってて。', 300, 300, 70, lt, tail=(120, 170))
    im.alpha_composite(vignette((40, 80, 220), 170))
    return im

def sc_sns(lt, dur, t, alt=False):
    c1, c2 = ((255, 120, 190), (130, 90, 255)) if not alt else ((90, 220, 255), (255, 120, 190))
    inner = radial(c1, c2, W / 2, H / 2, 1300)
    inner.alpha_composite(halftone((255, 255, 255), 30, 50))
    beat_ph = ((t - BEAT0) / (BEAT / 2)) % 1
    step = int((t - BEAT0) / (BEAT / 2)) % 2
    sw = math.sin(beat_ph * math.pi) * 10 * (1 if step else -1)
    bounce = -abs(math.sin(beat_ph * math.pi)) * 60
    put_char(inner, W / 2, 1880 + bounce, 1.15, angle=sw, flip=(step == 1) ^ alt, anchor='bottom')
    sparkles(inner, t, 40 + alt, 14)
    im = vgrad((40, 20, 60), (90, 30, 90))
    im.alpha_composite(focus_lines(70 + alt, (255, 160, 220), 60, 700, alpha=90))
    n = int(lerp(12000 if not alt else 98000, 98000 if not alt else 128000, lt / dur))
    phone_frame(im, inner, t, f'{n/10000:.1f}万')
    floating_hearts(im, lt, 9 + alt, 16, 230)
    return im

def sc_doki(lt, dur, t):
    im = radial((255, 230, 240), (255, 90, 150), W / 2, H * 0.45, 1100)
    im.alpha_composite(focus_lines(81, (255, 255, 255), 140, 420, alpha=180))
    z = lerp(4.2, 4.6, lt / dur)
    put_char(im, W / 2 + 40, 830, z, angle=-6, crop=(190, 10, 420, 200), blush=True)
    sfx(im, 'ドキッ', 780, 1330, 170, lt, fill=(255, 255, 255), stroke=HOTPINK)
    sparkles(im, t, 3, 20)
    return im

def sc_heart_climax(lt, dur, t):
    k = lt / dur
    im = radial((255, 255, 255), (255, 120, 180), W / 2, H * 0.45, lerp(700, 1500, k))
    lay = new(); d = ImageDraw.Draw(lay); heart(d, W / 2, 820, 520 * pop(k / 0.5, 1.2), fill=(255, 60, 130, 230), outline=(255, 255, 255), width=12); im.alpha_composite(lay)
    put_char(im, W / 2, 860, 1.25, crop=(0, 0, CW, 760), blush=True)
    heart_burst(im, k, 300, 40, W / 2, 820, 1100)
    return im

# --- warehouse rescue
def sc_cell(i):
    def f(lt, dur, t):
        im = bg_warehouse().copy()
        shake = (math.sin(lt * 80) * 18 * max(0, 1 - lt / 0.25)) if i > 0 else 0
        door(im, 0, crack=[0.0, 0.3, 0.6, 1.0][i])
        d = ImageDraw.Draw(im); rope(d, 1530)
        im.alpha_composite(vignette((0, 0, 0), 235, 1000))
        if i == 0: monologue(im, '（だれか…こわい）', lt, y=1300)
        else: sfx(im, ['', 'ドン!', 'ドンッ!', 'バァンッ!!'][i], W / 2 + [0, -200, 180, 0][i], [0, 640, 700, 760][i], [0, 150, 165, 175][i], lt, fill=(255, 255, 255), stroke=(0, 0, 0))
        if shake: im = ImageChops.offset(im, int(shake), int(shake * 0.5))
        return im
    return f

def sc_doorway(lt, dur, t):
    im = bg_warehouse().copy(); door(im, 1, light=1.0)
    put_char(im, W / 2, 1220, 0.52, mode='silhouette', anchor='bottom')
    lay = new(); ImageDraw.Draw(lay).rectangle((0, 0, W, H), fill=(255, 250, 220, int(200 * max(0, 1 - lt / 0.25)))); im.alpha_composite(lay)
    bubble(im, '待たせたな。', W / 2, 1480, 72, lt - 0.1, tail=(0, -200))
    return im

def sc_bad_guys(lt, dur, t):
    im = bg_warehouse().copy(); im.alpha_composite(focus_lines(91, (255, 30, 30), 110, 600, alpha=120))
    lay = new(); d = ImageDraw.Draw(lay)
    for i, (x, s) in enumerate([(230, 1.45), (W / 2, 1.75), (850, 1.45)]): thug(d, x, 1900, s, col=(20, 8, 10), lean=0.04 * (1 - i))
    im.alpha_composite(lay); im.alpha_composite(vignette((160, 0, 0), 210))
    bubble(im, 'なんだテメェ!!', W / 2, 330, 84, lt, jagged=True)
    return im

def sc_glint(lt, dur, t):
    im = new((5, 5, 12, 255)); im.alpha_composite(speed_lines(5, 0, (60, 70, 110), 50, 160))
    put_char(im, W / 2, 930, 3.6, crop=(200, 0, 420, 215), mode='dark')
    ex, ey = W / 2 + (EYE_R[0] - 310) * 3.6, 930 + (EYE_R[1] - 107) * 3.6
    k = pop(lt / 0.3, 1.4); lay = new(); d = ImageDraw.Draw(lay)
    sparkle(d, ex, ey, 260 * k, (255, 255, 255, 255)); sparkle(d, ex, ey, 120 * k, (255, 230, 120, 255))
    im.alpha_composite(lay.filter(ImageFilter.GaussianBlur(1)))
    text(im, 'ギンッ', 300, 1450, 140, fill=(255, 255, 255), stroke=(40, 60, 140), sw=12, scale=pop(lt / 0.3), bold=3, angle=6)
    return im

def sc_kick(lt, dur, t, ang=95, flip=False, txt='バキッ!!', bgc=((255, 255, 255), (255, 170, 40)), sl=0):
    im = radial(*bgc, W / 2, H / 2, 1100)
    im.alpha_composite(speed_lines(60 + sl, sl, (40, 20, 0), 80, 200))
    k = ease_out(lt / (dur * 0.45)); sgn = -1 if flip else 1
    x = lerp(W / 2 - sgn * 800, W / 2 - sgn * 100, k); y = 980
    ghost_trail(im, x, y, 0.62, ang, -sgn * 140, 0, 4, flip, 'mono')
    put_char(im, x, y, 0.62, ang, flip)
    tl = new(); td = ImageDraw.Draw(tl); thug(td, W / 2 + sgn * 330, 1600, 1.05, col=(15, 5, 5), lean=sgn * 0.25, bat=False)
    if lt > dur * 0.4:
        tl = tl.rotate(-sgn * 25 * ease_out((lt - dur * 0.4) / 0.2), center=(W / 2 + sgn * 330, 1500))
        paste(im, tl, sgn * 160 * ease_out((lt - dur * 0.4) / 0.25), 0)
        lay = new(); d = ImageDraw.Draw(lay); cx = W / 2 + sgn * 330; sparkle(d, cx, 950, 260, (255, 255, 255, 230)); im.alpha_composite(lay)
        sfx(im, txt, W / 2, 420, 200, lt - dur * 0.4, fill=(255, 40, 20), stroke=(255, 255, 255))
    else: im.alpha_composite(tl)
    return im

def sc_ko(lt, dur, t):
    im = bg_warehouse().copy(); door(im, 1, light=0.6)
    lay = new(); d = ImageDraw.Draw(lay)
    for i, x in enumerate([200, 850, 520]):
        tl = new(); td = ImageDraw.Draw(tl); thug(td, x, 1800, 0.8, col=(15, 10, 12), bat=False, eye=(90, 90, 90))
        tl = tl.rotate([88, -92, 85][i], center=(x, 1640)); lay.alpha_composite(tl)
    im.alpha_composite(lay)
    put_char(im, W / 2 + 40, 1640, 0.95, anchor='bottom')
    dust = new(); dd = ImageDraw.Draw(dust); rnd = random.Random(4)
    for _ in range(30):
        x = rnd.uniform(0, W); y = rnd.uniform(1400, 1800) - lt * rnd.uniform(40, 160); r = rnd.uniform(30, 90)
        dd.ellipse((x - r, y - r, x + r, y + r), fill=(200, 190, 170, 40))
    im.alpha_composite(dust.filter(ImageFilter.GaussianBlur(10)))
    text(im, 'ぐはっ…', 220, 1820, 70, fill=(255, 255, 255), stroke=(0, 0, 0), sw=8, angle=8)
    text(im, 'K.O.', W / 2, 380, 170, fill=(255, 220, 40), stroke=(120, 0, 0), sw=14, scale=pop(lt / 0.3, 1.3), bold=4, angle=-6)
    return im

def sc_untie(lt, dur, t):
    im = bg_bokeh((255, 230, 190), (200, 130, 110), 8)
    z = lerp(1.0, 1.1, ease_in_out(lt / dur))
    put_char(im, 560, 1180, 2.0 * z, angle=-10, crop=(0, 0, CW, 760), mode='warm')
    d = ImageDraw.Draw(im); rope(d, 1560, lt / dur, cut=True)
    bubble(im, 'ケガ、ない？', 780, 330, 76, lt - 0.05, tail=(-150, 160))
    if lt > dur * 0.5: monologue(im, '（来てくれた…）', lt - dur * 0.5, y=1420)
    sparkles(im, t, 33, 18, color=(255, 245, 210))
    im.alpha_composite(vignette((255, 140, 120), 140))
    return im

def sc_laugh(lt, dur, t):
    im = bg_bokeh((255, 245, 225), (255, 160, 170), 5)
    im.alpha_composite(focus_lines(93, (255, 255, 255), 90, 560, alpha=150))
    bob = math.sin(lt * 22) * 10 * max(0, 1 - lt / dur)  # laughing shake
    put_char(im, 560, 1080 + bob, 1.85, angle=-4, crop=(0, 0, CW, 760), smile=True, blush=True)
    hx = 560 + (HAND[0] - CW / 2) * 1.85; hy = 1080 + (HAND[1] - 380) * 1.85
    emoji(im, '👍', hx, hy, 240 * pop((lt - 0.3) / 0.3, 1.3) if lt > 0.3 else 1, angle=12, alpha=1 if lt > 0.3 else 0)
    sfx(im, 'ハハッ', 830, 420, 130, lt, fill=(255, 240, 120), stroke=(180, 70, 0), angle=8)
    if lt > 0.45: bubble(im, 'もう大丈夫。', 330, 1420, 66, lt - 0.45, tail=(140, -170))
    floating_hearts(im, lt, 14, 14)
    sparkles(im, t, 44, 20)
    im.alpha_composite(vignette((255, 90, 160), 150))
    return im

def sc_outro(lt, dur, t):
    im = bg_sunset().copy()
    z = lerp(1.0, 1.12, ease_in_out(lt / dur))
    put_char(im, W / 2 + 30, 1960, 1.08 * z, anchor='bottom', mode='warm', smile=lt > dur * 0.45)
    floating_hearts(im, lt, 21, 12, 180)
    sparkles(im, t, 51, 14, (0, 0, W, 1200), (255, 250, 220))
    im.alpha_composite(vignette((255, 90, 140), 160))
    if lt > 0.6: text(im, '── ハトリーマン ──', W / 2, 140, 66, fill=(255, 255, 255), stroke=(200, 60, 100), sw=7, alpha=min(1, (lt - 0.6) / 0.4), bold=1)
    return im

SCENES = [  # (start, end, fn)
    (0.0, B(2), sc_commute),
    (B(2), B(4), sc_thumbs),
    (B(4), B(5), make_chu(0)), (B(5), B(6), make_chu(1)), (B(6), B(7), make_chu(2)), (B(7), B(8), make_chu(3)),
    (B(8), B(9), sc_office),
    (B(9), B(10), sc_umbrella),
    (B(10), B(12), sc_comfort),
    (B(12), B(13), sc_yankee),
    (B(13), B(14), sc_flykick),
    (B(14), B(15), sc_impact),
    (B(15), B(16), sc_guard),
    (B(16), B(17), sc_sns),
    (B(17), B(18), lambda lt, d, t: sc_sns(lt, d, t, True)),
    (B(18), B(19), sc_doki),
    (B(19), B(20), sc_heart_climax),
    (B(20), B(21), sc_cell(0)), (B(21), B(22), sc_cell(1)), (B(22), B(23), sc_cell(2)), (B(23), B(24), sc_cell(3)),
    (B(24), B(25), sc_doorway),
    (B(25), B(26), sc_bad_guys),
    (B(26), B(27), sc_glint),
    (B(27), B(28), sc_kick),
    (B(28), B(29), lambda lt, d, t: sc_impact(lt, d, t, 'ドガァ!!')),
    (B(29), B(30), lambda lt, d, t: sc_kick(lt, d, t, -95, True, 'ドゴォ!!', ((255, 255, 255), (80, 170, 255)), 12)),
    (B(30), B(31), lambda lt, d, t: sc_flykick(lt, d, t, False, 'ズドンッ!!', (255, 120, 20))),
    (B(31), B(32), sc_ko),
    (B(32), B(34), sc_untie),
    (B(34), B(36), sc_laugh),
    (B(36), DURATION + 0.1, sc_outro),
]
HIT_BEATS = {13, 14, 21, 22, 23, 27, 28, 29, 30}      # camera shake + chromatic split
FLASH_BEATS = {4, 20, 24, 36}

def chroma(im, s):
    r, g, b, a = im.split(); r = ImageChops.offset(r, s, 0); b = ImageChops.offset(b, -s, 0)
    return Image.merge('RGBA', (r, g, b, a))

@lru_cache(1)
def grain_tiles():
    rnd = np.random.default_rng(1)
    return [Image.fromarray(rnd.integers(0, 255, (H // 2, W // 2), dtype=np.uint8)).resize((W, H)) for _ in range(4)]

def render(fi):
    t = fi / FPS
    for (s, e, fn) in SCENES:
        if s <= t < e: im = fn(t - s, e - s, t); break
    else: im = SCENES[-1][2](t - SCENES[-1][0], 1, t)
    bi = int(math.floor((t - BEAT0) / BEAT)); bp = (t - BEAT0) / BEAT - bi
    lyric_telop(im, t)
    # beat zoom punch
    z = 1 + 0.045 * math.exp(-bp * 6) if t > BEAT0 else 1
    if bi in HIT_BEATS: z += 0.04 * math.exp(-bp * 5)
    if z > 1.001:
        cw, ch = W / z, H / z; im = im.crop(((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)).resize((W, H), Image.BILINEAR)
    if bi in HIT_BEATS and bp < 0.45:
        k = 1 - bp / 0.45; im = chroma(im, int(14 * k)); im = ImageChops.offset(im, int(math.sin(t * 95) * 22 * k), int(math.cos(t * 83) * 16 * k))
    if bi in FLASH_BEATS and bp < 0.2:
        im = Image.blend(im, new((255, 255, 255, 255)), 0.85 * (1 - bp / 0.2))
    if t > DURATION - 0.45:
        im = Image.blend(im, new((255, 255, 255, 255)), (t - (DURATION - 0.45)) / 0.45)
    if t < 0.12: im = Image.blend(new((255, 255, 255, 255)), im, t / 0.12)
    rgb = im.convert('RGB')
    g = grain_tiles()[fi % 4]
    rgb = Image.blend(rgb, Image.merge('RGB', (g, g, g)), 0.035)
    return rgb.tobytes()

def main():
    args = sys.argv[1:]
    if '--preview' in args:
        ts = [float(x) for x in args[args.index('--preview') + 1].split(',')]
        out = args[args.index('--preview') + 2]
        os.makedirs(out, exist_ok=True)
        for t in ts:
            fi = int(round(t * FPS)); Image.frombytes('RGB', (W, H), render(fi)).resize((W // 2, H // 2)).save(f'{out}/f_{t:05.2f}.jpg', quality=85)
        return
    out = args[0] if args else os.path.join(ROOT, 'output/hatoriman_mad.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = int(math.ceil(DURATION * FPS))
    ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-i', os.path.join(ROOT, 'assets/song.mp3'), '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                           '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as p:
        for i, fr in enumerate(p.imap(render, range(n), chunksize=4)):
            ff.stdin.write(fr)
            if i % 60 == 0: print(f'frame {i}/{n}', flush=True)
    ff.stdin.close(); ff.wait(); print('done', out)

if __name__ == '__main__':
    main()
