"""Auditable educational ORB-style pipeline, not OpenCV's trained ORB.

Images are grayscale float64 in [0,255], indexed [row, col]. Display (x,y)
means (col,row), with positive y down. FAST uses strict >/< comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from utils.harris import harris_response, tensor_maps

CIRCLE = ((-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3),
          (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
          (0, -3), (-1, -3), (-2, -2), (-3, -1))
CARDINAL = (0, 4, 8, 12)
PRESET_GOALS = {
    "Góc mạnh": "Cung FAST-12 và điểm số rõ ràng.",
    "Vùng phẳng": "Không có cung sáng/tối đủ dài.",
    "Cạnh ngang": "Biên thẳng không mặc nhiên là góc FAST-12.",
    "Bàn cờ": "Nhiều góc và cạnh tranh qua NMS.",
    "Cấu trúc bất đối xứng": "Centroid và so sánh ảnh quay.",
    "Hoa văn lặp": "Descriptor giống nhau có thể gây match nhầm.",
    "Cấu trúc đổi tỉ lệ": "Pyramid chỉ cho tính ổn định xấp xỉ.",
    "Cảnh có nhiễu": "Ngưỡng và Gaussian smoothing ảnh hưởng kết quả.",
}


@dataclass(frozen=True)
class FastResult:
    valid: bool
    center: float
    values: tuple[float, ...]
    labels: tuple[str, ...]
    early_pass: bool
    accepted: bool
    score: float
    arc: tuple[int, ...]  # zero-based indices in clockwise ring order
    polarity: str
    margins: tuple[float, ...]
    longest: int


@dataclass(frozen=True)
class Level:
    image: np.ndarray
    scale_x: float  # exact center-to-center coordinate ratio, W0/Wl
    scale_y: float
    nominal_scale: float


@dataclass(frozen=True)
class Keypoint:
    level: int
    row: int
    col: int
    x_original: float
    y_original: float
    fast_score: float
    harris_score: float
    angle: float | None = None
    descriptor: bytes | None = None


@dataclass(frozen=True)
class Moments:
    m00: float
    m10: float
    m01: float
    cx: float | None
    cy: float | None
    angle: float | None
    patch: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class Brief:
    bits: np.ndarray
    packed: bytes | None
    samples: tuple[tuple[float, float, float, float, float, float], ...]
    steering_angle: float
    reason: str = ""


@dataclass(frozen=True)
class Match:
    query: int
    train: int | None
    distance: int | None
    second_distance: int | None
    accepted: bool
    reason: str
    geometric_error: float | None = None
    correct_geometry: bool | None = None


def _image(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 7 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 255:
        raise ValueError("Expected finite grayscale image, >=7x7, intensity in [0,255]")
    return a


def get_fast_circle_offsets() -> tuple[tuple[int, int], ...]:
    return CIRCLE


def classify_fast_ring(values: np.ndarray, center: float, threshold: float) -> tuple[str, ...]:
    if threshold < 0 or len(values) != 16:
        raise ValueError("Need 16 ring values and nonnegative threshold")
    return tuple("BRIGHT" if v > center + threshold else
                 "DARK" if v < center - threshold else "SIMILAR" for v in values)


def evaluate_fast_early_test(labels: tuple[str, ...], arc_length: int = 12) -> bool:
    """Three of four cardinal pixels is necessary only for FAST-12."""
    if arc_length != 12:
        raise ValueError("Four-point >=3 pretest is only proved for FAST-12")
    return any(sum(labels[i] == side for i in CARDINAL) >= 3 for side in ("BRIGHT", "DARK"))


def _best_arc(values: np.ndarray, center: float, length: int) -> tuple[float, tuple[int, ...], str, tuple[float, ...]]:
    """Maximize minimum positive margin; ties: BRIGHT then lowest start."""
    best = (0.0, (), "", ())
    for polarity in ("BRIGHT", "DARK"):
        margins = values - center if polarity == "BRIGHT" else center - values
        for start in range(16):
            ids = tuple((start + j) % 16 for j in range(length))
            vals = tuple(float(margins[i]) for i in ids)
            score = min(vals)
            if score > best[0]:
                best = (float(score), ids, polarity, vals)
    return best


def detect_fast_corner(image: np.ndarray, row: int, col: int, threshold: float = 20,
                       arc_length: int = 12, early: bool = True) -> FastResult:
    a = _image(image)
    if threshold < 0 or not 1 <= arc_length <= 16:
        raise ValueError("Invalid FAST threshold or arc length")
    if not 3 <= row < a.shape[0]-3 or not 3 <= col < a.shape[1]-3:
        return FastResult(False, float("nan"), (), (), False, False, 0, (), "", (), 0)
    center = float(a[row, col])
    values = np.array([a[row+dr, col+dc] for dr, dc in CIRCLE])
    labels = classify_fast_ring(values, center, threshold)
    early_pass = evaluate_fast_early_test(labels) if arc_length == 12 else False
    score, arc, polarity, margins = _best_arc(values, center, arc_length)
    accepted = bool(score > threshold)
    if arc_length == 12 and early and not early_pass:
        assert not accepted  # necessary condition; catches an implementation regression
    longest = 0
    for side in ("BRIGHT", "DARK"):
        flags = [s == side for s in labels]
        longest = max(longest, max((sum(flags[(i+j) % 16] for j in range(k)) == k and k
                                    for i in range(16) for k in range(1, 17)), default=0))
    return FastResult(True, center, tuple(map(float, values)), labels, early_pass,
                      accepted, score, arc, polarity, margins, int(longest))


def fast_score_map(image: np.ndarray, arc_length: int = 12) -> np.ndarray:
    """Vectorized exact max-min arc margin on valid circle centers only."""
    a = _image(image)
    if not 1 <= arc_length <= 16:
        raise ValueError("arc length must be 1..16")
    result = np.zeros(a.shape, dtype=np.float64)
    center = a[3:-3, 3:-3]
    ring = np.stack([a[3+dr:a.shape[0]-3+dr, 3+dc:a.shape[1]-3+dc] for dr, dc in CIRCLE])
    bright, dark = ring-center, center-ring
    for polarity in (bright, dark):
        for start in range(16):
            margin = np.minimum.reduce([polarity[(start+j) % 16] for j in range(arc_length)])
            np.maximum(result[3:-3, 3:-3], margin, out=result[3:-3, 3:-3])
    return result


def detect_fast_candidates(image: np.ndarray, threshold: float = 20,
                           arc_length: int = 12) -> tuple[np.ndarray, np.ndarray]:
    scores = fast_score_map(image, arc_length)
    return scores > threshold, scores


def apply_fast_nms(scores: np.ndarray, eligible: np.ndarray, size: int = 3) -> np.ndarray:
    """Keep strict local maximum, or lexicographically first tied (row,col)."""
    if scores.shape != eligible.shape or size < 3 or size % 2 != 1:
        raise ValueError("Invalid NMS inputs")
    radius = size // 2
    out = np.zeros_like(eligible, dtype=bool)
    for row, col in np.argwhere(eligible):
        if row < radius or col < radius or row >= scores.shape[0]-radius or col >= scores.shape[1]-radius:
            continue
        window = scores[row-radius:row+radius+1, col-radius:col+radius+1]
        allowed = eligible[row-radius:row+radius+1, col-radius:col+radius+1]
        maximum = float(np.max(window[allowed]))
        winners = np.argwhere(allowed & (window == maximum))
        out[row,col] = scores[row,col] == maximum and tuple(winners[0]) == (radius,radius)
    return out


def rank_fast_candidates_with_harris(image: np.ndarray, nms: np.ndarray,
                                     scores: np.ndarray, limit: int = 50) -> tuple[list[tuple[int,int,float,float]], np.ndarray]:
    """Harris ranks only surviving FAST+NMS points; it never grants eligibility."""
    if limit < 0 or nms.shape != scores.shape:
        raise ValueError("Invalid ranking inputs")
    response = harris_response(tensor_maps(_image(image)/255, size=5, sigma=1.2))
    candidates = [(int(r),int(c),float(scores[r,c]),float(response[r,c])) for r,c in np.argwhere(nms)]
    candidates.sort(key=lambda v:(-v[3],-v[2],v[0],v[1]))
    return candidates[:limit], response


def build_orb_image_pyramid(image: np.ndarray, levels: int = 8,
                            scale_factor: float = 1.2) -> tuple[Level, ...]:
    a = _image(image)
    if not 1 <= levels <= 10 or not 1.05 <= scale_factor <= 2:
        raise ValueError("Unsafe pyramid parameters")
    out=[]
    for level in range(levels):
        nominal=scale_factor**level
        width=max(7, int(round(a.shape[1]/nominal)))
        height=max(7, int(round(a.shape[0]/nominal)))
        if width == a.shape[1] and height == a.shape[0]:
            small=a.copy()
        else:
            # Explicit anti-alias prior to INTER_AREA resampling from original.
            sigma=max(.4, .5*nominal)
            blur=cv2.GaussianBlur(a,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REFLECT_101)
            small=cv2.resize(blur,(width,height),interpolation=cv2.INTER_AREA)
        out.append(Level(small,a.shape[1]/width,a.shape[0]/height,nominal))
    return tuple(out)


def map_pyramid_coordinates(level: Level, row: float, col: float) -> tuple[float,float]:
    """OpenCV resize pixel-center mapping: x0=(x+.5)*W0/Wl-.5."""
    return ((col+.5)*level.scale_x-.5,(row+.5)*level.scale_y-.5)


def calculate_intensity_moments(image: np.ndarray, row: int, col: int,
                                radius: int = 9) -> Moments:
    a=_image(image)
    if radius < 1 or row-radius < 0 or col-radius < 0 or row+radius >= a.shape[0] or col+radius >= a.shape[1]:
        raise ValueError("Circular orientation patch crosses image border")
    patch=a[row-radius:row+radius+1,col-radius:col+radius+1]
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    mask=xx*xx+yy*yy<=radius*radius
    m00=float(np.sum(patch[mask]))
    m10=float(np.sum((xx*patch)[mask]))
    m01=float(np.sum((yy*patch)[mask]))
    norm=math.hypot(m10,m01)
    if m00 <= 1e-12 or norm <= max(1e-12,1e-9*m00):
        return Moments(m00,m10,m01,None,None,None,patch,mask)
    return Moments(m00,m10,m01,m10/m00,m01/m00,math.degrees(math.atan2(m01,m10))%360,patch,mask)


def generate_brief_test_pattern(n: int = 256, radius: int = 7, seed: int = 2026) -> np.ndarray:
    if n < 1 or n > 512 or radius < 2:
        raise ValueError("Invalid BRIEF pattern configuration")
    coords=np.array([(x,y) for y in range(-radius,radius+1) for x in range(-radius,radius+1)
                     if x*x+y*y<=radius*radius],dtype=np.int16)
    rng=np.random.default_rng(seed)
    pairs=[]
    while len(pairs)<n:
        p,q=rng.integers(0,len(coords),size=2)
        if p!=q:
            pairs.append((*coords[p],*coords[q]))
    return np.asarray(pairs,dtype=np.float64)


def quantize_orientation(angle: float, bins: int = 30) -> float:
    if bins < 1 or not np.isfinite(angle):
        raise ValueError("Invalid orientation")
    width=360/bins
    return (math.floor((angle%360)/width+.5)%bins)*width


def steer_brief_pairs(pattern: np.ndarray, angle: float, quantized: bool = True) -> tuple[np.ndarray,float]:
    pairs=np.asarray(pattern,dtype=np.float64)
    if pairs.ndim!=2 or pairs.shape[1]!=4:
        raise ValueError("Expected n×4 BRIEF pattern")
    used=quantize_orientation(angle) if quantized else angle%360
    radians=math.radians(used)
    rotation=np.array([[math.cos(radians),-math.sin(radians)],
                       [math.sin(radians), math.cos(radians)]])
    return (pairs.reshape(-1,2)@rotation.T).reshape(-1,4),used


def sample_brief_pair_intensities(image: np.ndarray, row: int, col: int,
                                  pair: np.ndarray) -> tuple[float,float,float,float,float,float] | None:
    """Nearest-neighbor after np.rint; invalid coordinates reject the pair."""
    a=_image(image)
    x1,y1,x2,y2=map(float,pair)
    px,py,qx,qy=map(lambda z:int(np.rint(z)),(col+x1,row+y1,col+x2,row+y2))
    if not (0<=px<a.shape[1] and 0<=qx<a.shape[1] and 0<=py<a.shape[0] and 0<=qy<a.shape[0]):
        return None
    return (float(px),float(py),float(qx),float(qy),float(a[py,px]),float(a[qy,qx]))


def pack_binary_descriptor(bits: np.ndarray) -> bytes:
    b=np.asarray(bits)
    if b.shape!=(256,) or not np.isin(b,(0,1)).all():
        raise ValueError("A complete 256-bit descriptor is required")
    return np.packbits(b.astype(np.uint8),bitorder="big").tobytes()


def unpack_binary_descriptor(data: bytes) -> np.ndarray:
    if not isinstance(data,bytes) or len(data)!=32:
        raise ValueError("Expected 32 bytes")
    return np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder="big")


def build_oriented_brief_descriptor(image: np.ndarray, row: int, col: int,
                                    angle: float | None, pattern: np.ndarray | None = None,
                                    steering: bool = True, quantized: bool = True) -> Brief:
    pairs=generate_brief_test_pattern() if pattern is None else np.asarray(pattern)
    if pairs.ndim!=2 or pairs.shape[1]!=4:
        raise ValueError("Expected n×4 pattern")
    if angle is None or not np.isfinite(angle):
        return Brief(np.empty(0,dtype=np.uint8),None,(),0,"undefined orientation")
    rotated,used=steer_brief_pairs(pairs,angle if steering else 0,quantized and steering)
    a=_image(image)
    smoothed=cv2.GaussianBlur(a,(0,0),sigmaX=1.2,sigmaY=1.2,borderType=cv2.BORDER_REFLECT_101)
    samples=[];bits=[]
    for pair in rotated:
        values=sample_brief_pair_intensities(smoothed,row,col,pair)
        if values is None:
            return Brief(np.empty(0,dtype=np.uint8),None,tuple(samples),used,"sample outside image")
        samples.append(values)
        bits.append(int(values[4]<values[5]))
    bit_array=np.asarray(bits,dtype=np.uint8)
    packed=pack_binary_descriptor(bit_array) if len(bits)==256 else None
    return Brief(bit_array,packed,tuple(samples),used)


def bit_entropy(q: float) -> float:
    if not 0<=q<=1:
        raise ValueError("q must be a probability")
    return 0. if q in (0,1) else -q*math.log2(q)-(1-q)*math.log2(1-q)


def pattern_learning_demo() -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Small fixed synthetic training set; educational selection, not ORB pattern."""
    rng=np.random.default_rng(153)
    patches=[]
    yy,xx=np.mgrid[-9:10,-9:10]
    for _ in range(40):
        angle=rng.uniform(0,2*math.pi)
        offset=rng.uniform(-3,3)
        plane=(xx*math.cos(angle)+yy*math.sin(angle)>offset).astype(float)*150+45
        patches.append(np.clip(plane+rng.normal(0,12,plane.shape),0,255))
    candidate=generate_brief_test_pattern(80,radius=6,seed=81)
    bits=np.array([[int(patch[9+int(pair[1]),9+int(pair[0])]<
                        patch[9+int(pair[3]),9+int(pair[2])]) for pair in candidate]
                   for patch in patches],dtype=np.uint8)
    q=bits.mean(axis=0)
    entropy=np.array([bit_entropy(float(v)) for v in q])
    selected=[]
    for idx in np.argsort(-entropy,kind="stable"):
        if not selected or all(abs(float(np.corrcoef(bits[:,idx],bits[:,j])[0,1]))<.65
                               for j in selected if np.std(bits[:,j])>0):
            selected.append(int(idx))
        if len(selected)==12:break
    return candidate,bits,entropy,np.asarray(selected,dtype=int)


def calculate_hamming_distance(a: bytes, b: bytes) -> int:
    if not isinstance(a,bytes) or not isinstance(b,bytes) or len(a)!=32 or len(b)!=32:
        raise ValueError("Hamming requires two valid 32-byte descriptors")
    return sum((x^y).bit_count() for x,y in zip(a,b))


def match_binary_descriptors(a: list[Keypoint], b: list[Keypoint],
                             method: str = "cross-check", ratio: float = .8,
                             transform: np.ndarray | None = None,
                             tolerance: float = 4.) -> list[Match]:
    if method not in ("nearest","cross-check","ratio") or not 0<ratio<1:
        raise ValueError("Invalid matching rule")
    valid_b=[(j,p) for j,p in enumerate(b) if p.descriptor is not None]
    reverse={}
    if method=="cross-check":
        valid_a=[(i,p) for i,p in enumerate(a) if p.descriptor is not None]
        for j,p in valid_b:
            reverse[j]=min(((calculate_hamming_distance(p.descriptor,q.descriptor),i)
                            for i,q in valid_a),default=(math.inf,None))[1]
    matches=[]
    for i,p in enumerate(a):
        if p.descriptor is None or not valid_b:
            matches.append(Match(i,None,None,None,False,"missing descriptor"));continue
        ranked=sorted((calculate_hamming_distance(p.descriptor,q.descriptor),j)
                      for j,q in valid_b)
        d,j=ranked[0];d2=ranked[1][0] if len(ranked)>1 else None
        if method=="ratio":
            accept=d2 is not None and d2>0 and d/d2<ratio
            reason="ratio pass" if accept else "ratio fail / insufficient neighbors"
        elif method=="cross-check":
            accept=reverse[j]==i;reason="mutual nearest" if accept else "not mutual"
        else:
            accept=True;reason="nearest"
        error=None;correct=None
        if transform is not None:
            xy=np.asarray(transform,dtype=float)@np.array([p.x_original,p.y_original,1.])
            error=float(np.hypot(xy[0]-b[j].x_original,xy[1]-b[j].y_original))
            correct=error<=tolerance
        matches.append(Match(i,j,d,d2,accept,reason,error,correct))
    return matches


def teaching_presets(size: int = 129) -> dict[str,np.ndarray]:
    if size<65 or size%2!=1:raise ValueError("Use odd size >=65")
    yy,xx=np.indices((size,size));m=size//2
    corner=((xx>=m)&(yy>=m)).astype(float)*200+25
    # Embed one exact FAST-12 teaching center in the large corner scene.
    corner[m,m]=100
    for i,(dr,dc) in enumerate(CIRCLE):corner[m+dr,m+dc]=210 if i<12 else 100
    checker=(((xx//13)+(yy//13))%2).astype(float)*190+30
    asym=np.full((size,size),30.,dtype=float)
    cv2.rectangle(asym,(22,19),(64,71),190,-1)
    cv2.circle(asym,(87,42),13,245,-1)
    cv2.line(asym,(55,93),(105,107),160,6)
    cv2.circle(asym,(35,95),8,105,-1)
    repeat=np.full_like(asym,35)
    for x in (24,58,92):
        for y in (27,65,101):cv2.rectangle(repeat,(x-8,y-8),(x+8,y+8),190,-1)
    scaled=cv2.warpAffine(asym,cv2.getRotationMatrix2D((m,m),0,.78),(size,size),
                          flags=cv2.INTER_LINEAR,borderValue=30)
    rng=np.random.default_rng(2026)
    noisy=np.clip(asym+rng.normal(0,22,asym.shape),0,255)
    return dict(zip(PRESET_GOALS, (corner,np.full_like(corner,100),
                np.where(yy>=m,220.,30.),checker,asym,repeat,scaled,noisy)))


def microscope_fixtures() -> dict[str,np.ndarray]:
    """Pixel-exact 15×15 FAST examples at (7,7), no fabricated labels."""
    out={}
    for name,bright in (("Cung 12",list(range(12))),
                        ("Wrap 16→1",list(range(12,16))+list(range(8))),
                        ("Pretest sống, cung rớt",[0,4,8]),
                        ("Early reject",[0,4])):
        a=np.full((15,15),100.,dtype=float)
        for i in bright:
            dr,dc=CIRCLE[i];a[7+dr,7+dc]=210
        out[name]=a
    out["Vùng phẳng"]=np.full((15,15),100.)
    out["Cạnh thẳng"]=np.where(np.indices((15,15))[0]>=7,210.,40.)
    return out


def transform_scene(image: np.ndarray, angle: float = 20, scale: float = 1.,
                    brightness: float = 0, contrast: float = 1., noise: float = 0) -> tuple[np.ndarray,np.ndarray]:
    a=_image(image);h,w=a.shape
    matrix=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),-angle,scale)
    warped=cv2.warpAffine(a,matrix,(w,h),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=30)
    rng=np.random.default_rng(2026)
    return np.clip(warped*contrast+brightness+rng.normal(0,noise,warped.shape),0,255),matrix


def orb_pipeline(image: np.ndarray, *, threshold: float = 20, levels: int = 8,
                 scale_factor: float = 1.2, max_per_level: int = 30,
                 describe: bool = True) -> tuple[tuple[Level,...],list[Keypoint],list[dict]]:
    pyramid=build_orb_image_pyramid(image,levels,scale_factor)
    pattern=generate_brief_test_pattern()
    points=[];diagnostics=[]
    for index,level in enumerate(pyramid):
        eligible,scores=detect_fast_candidates(level.image,threshold)
        nms=apply_fast_nms(scores,eligible)
        ranked,harris=rank_fast_candidates_with_harris(level.image,nms,scores,max_per_level)
        diagnostics.append(dict(eligible=eligible,scores=scores,nms=nms,harris=harris,ranked=ranked))
        for row,col,score,response in ranked:
            try: moments=calculate_intensity_moments(level.image,row,col)
            except ValueError: continue
            if moments.angle is None:continue
            brief=build_oriented_brief_descriptor(level.image,row,col,moments.angle,pattern) if describe else None
            xy=map_pyramid_coordinates(level,row,col)
            points.append(Keypoint(index,row,col,*xy,score,response,moments.angle,
                                   brief.packed if brief is not None else None))
    return pyramid,points,diagnostics
