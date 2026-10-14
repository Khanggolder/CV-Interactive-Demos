"""Inspectable SIFT-style teaching engine, independent of Streamlit.

Images are finite grayscale float64 in [0,1], indexed [row, col]. The local
Taylor coordinate order is (x=col, y=row, s=DoG index); image y points down.
This is a documented discrete approximation, not OpenCV/Lowe bit equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, cos, radians, sin, sqrt

import cv2
import numpy as np

from utils.scale_space import Octave, build_pyramid, detect_extrema, presets as scale_presets


@dataclass(frozen=True)
class Keypoint:
    octave: int
    layer: int
    row: int
    col: int
    dog_value: float
    kind: str
    row_local: float
    col_local: float
    layer_local: float
    row_global: float
    col_global: float
    sigma_local: float
    sigma_global: float
    response: float
    status: str = "raw"
    reason: str = ""
    delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: float | None = None
    descriptor: np.ndarray | None = None


@dataclass(frozen=True)
class Taylor:
    gradient: np.ndarray  # Dx, Dy, Ds
    hessian: np.ndarray   # x, y, s order
    center: float


@dataclass(frozen=True)
class Orientation:
    histogram: np.ndarray
    raw_histogram: np.ndarray
    contributions: tuple[tuple[int, int, float, float, float, float], ...]
    peaks: tuple[float, ...]
    gaussian_layer: int


@dataclass(frozen=True)
class Descriptor:
    raw: np.ndarray
    first_l2: np.ndarray
    clipped: np.ndarray
    final: np.ndarray
    cells: np.ndarray
    samples: tuple[tuple[int, int, float, float, float, float], ...]


@dataclass(frozen=True)
class Match:
    query: int
    nearest: int | None
    second: int | None
    d1: float | None
    d2: float | None
    ratio: float | None
    accepted: bool
    reason: str


def teaching_presets(size: int = 129) -> dict[str, np.ndarray]:
    """Deterministic synthetic scenes; labels are teaching goals, not detections."""
    base = scale_presets(size)
    yy, xx = np.indices((size, size))
    c = size // 2
    disk = lambda x, y, r: ((xx-x)**2 + (yy-y)**2 <= r*r).astype(float)
    scene = np.clip(.12 + .72*disk(c-32,c-30,7) + .65*disk(c+24,c-17,12)
                    + .8*disk(c-18,c+29,5) + .55*disk(c+27,c+26,9)
                    + .32*((xx>c-7)&(xx<c+8)&(yy>c-8)&(yy<c+9)), 0, 1)
    repeat = np.full((size,size), .16)
    for y in (28, 54, 80, 106):
        for x in (28, 54, 80, 106):
            repeat += .72*disk(x,y,5)
    weak = np.clip(.36 + .09*disk(c,c,8) + .65*disk(c-27,c+22,10), 0, 1)
    asymmetric = np.clip(.12 + .8*disk(c-20,c-18,8) + .56*disk(c+26,c+8,14)
                         + .6*((xx>c-7)&(xx<c+6)&(yy>c-5)&(yy<c+5)), 0, 1)
    rng = np.random.default_rng(2026)
    return {
        "Blob và góc rõ": scene,
        "Blob đa thang": base["Ba blob sáng"],
        "Cấu trúc bất đối xứng": asymmetric,
        "Tương phản yếu": weak,
        "Cạnh kéo dài": np.clip(.12+.8*(xx>c)+.13*disk(c+19,c+18,7),0,1),
        "Hoa văn lặp": np.clip(repeat,0,1),
        "Cảnh biến đổi affine": scene.copy(),
        "Cảnh nhiễu": np.clip(scene+rng.normal(0,.10,scene.shape),0,1),
    }


def detect_dog_candidates(pyramid: tuple[Octave, ...]) -> list[Keypoint]:
    """Reuse strict 26-neighbor extrema from scale_space; no surrogate detector."""
    result = []
    for octave in pyramid:
        for layer,row,col,value,kind in detect_extrema(octave.dogs):
            factor = 2**octave.index
            result.append(Keypoint(octave.index,layer,row,col,value,kind,
                                   float(row),float(col),float(layer),
                                   float(row*factor),float(col*factor),
                                   octave.local_sigmas[layer],octave.global_sigmas[layer],value))
    return result


def calculate_dog_derivatives(stack: tuple[np.ndarray,...] | list[np.ndarray],
                              layer: int, row: int, col: int) -> Taylor:
    """Unit-step centered differences, including mixed /4, in (x,y,s) order."""
    if not 1 <= layer < len(stack)-1 or not 1 <= row < stack[layer].shape[0]-1 or not 1 <= col < stack[layer].shape[1]-1:
        raise ValueError("Need complete 3x3x3 interior")
    d = lambda s,y,x: float(stack[s][y,x])
    c = d(layer,row,col)
    dx = (d(layer,row,col+1)-d(layer,row,col-1))/2
    dy = (d(layer,row+1,col)-d(layer,row-1,col))/2
    ds = (d(layer+1,row,col)-d(layer-1,row,col))/2
    dxx = d(layer,row,col+1)+d(layer,row,col-1)-2*c
    dyy = d(layer,row+1,col)+d(layer,row-1,col)-2*c
    dss = d(layer+1,row,col)+d(layer-1,row,col)-2*c
    dxy = (d(layer,row+1,col+1)-d(layer,row+1,col-1)
           -d(layer,row-1,col+1)+d(layer,row-1,col-1))/4
    dxs = (d(layer+1,row,col+1)-d(layer+1,row,col-1)
           -d(layer-1,row,col+1)+d(layer-1,row,col-1))/4
    dys = (d(layer+1,row+1,col)-d(layer+1,row-1,col)
           -d(layer-1,row+1,col)+d(layer-1,row-1,col))/4
    g = np.array([dx,dy,ds])
    h = np.array([[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]])
    return Taylor(g,h,c)


def refine_keypoint(candidate: Keypoint, octave: Octave, max_iterations: int = 5,
                    max_condition: float = 1e12) -> Keypoint:
    """Move any offset >= .5 one lattice unit, then solve again; failures stay rejected."""
    layer,row,col = candidate.layer,candidate.row,candidate.col
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    for _ in range(max_iterations):
        try:
            t = calculate_dog_derivatives(octave.dogs,layer,row,col)
            if not np.isfinite(t.gradient).all() or not np.isfinite(t.hessian).all():
                return replace(candidate,status="invalid",reason="Đạo hàm không hữu hạn")
            if np.linalg.cond(t.hessian) > max_condition:
                return replace(candidate,status="invalid",reason="Hessian 3×3 suy biến hoặc điều kiện xấu")
            delta = -np.linalg.solve(t.hessian,t.gradient)
        except (ValueError, np.linalg.LinAlgError):
            return replace(candidate,status="invalid",reason="Hessian 3×3 suy biến hoặc vượt biên")
        if not np.isfinite(delta).all():
            return replace(candidate,status="invalid",reason="Độ lệch không hữu hạn")
        moves = np.where(np.abs(delta)>=.5,np.sign(delta).astype(int),0)
        if not np.any(moves):
            x,y,s = delta
            factor = 2**candidate.octave
            k = 2**(1/(len(octave.gaussians)-3))
            sigma = octave.local_sigmas[0]*k**(layer+s)
            response = t.center+.5*float(t.gradient@delta)
            return replace(candidate,layer=layer,row=row,col=col,
                           col_local=col+x,row_local=row+y,layer_local=layer+s,
                           col_global=(col+x)*factor,row_global=(row+y)*factor,
                           sigma_local=sigma,sigma_global=sigma*factor,
                           response=response,delta=(float(x),float(y),float(s)),
                           status="refined",reason="Hội tụ")
        col += int(moves[0]); row += int(moves[1]); layer += int(moves[2])
        if not 1 <= layer < len(octave.dogs)-1 or not 1 <= row < octave.dogs[0].shape[0]-1 or not 1 <= col < octave.dogs[0].shape[1]-1:
            return replace(candidate,status="invalid",reason="Dịch chuyển ra ngoài vùng 3×3×3 hợp lệ")
    return replace(candidate,status="invalid",reason="Không hội tụ trong số vòng lặp cho phép")


def calculate_spatial_hessian(stack: tuple[np.ndarray,...], layer: int,
                              row: int, col: int) -> np.ndarray:
    return calculate_dog_derivatives(stack,layer,row,col).hessian[:2,:2].copy()


def edge_decision(hxy: np.ndarray, edge_ratio: float = 10.0) -> tuple[bool,float,float,float]:
    """Keep only det>0 and trace²/det < (r+1)²/r (strict boundary)."""
    if not np.isfinite(edge_ratio) or edge_ratio <= 0:
        raise ValueError("edge_ratio must be positive")
    h = np.asarray(hxy,dtype=float)
    if h.shape != (2,2) or not np.isfinite(h).all():
        return False,float("nan"),float("nan"),float("nan")
    trace = float(np.trace(h)); det = float(np.linalg.det(h))
    ratio = trace*trace/det if det > 1e-15 else float("inf")
    boundary=(edge_ratio+1)**2/edge_ratio
    return bool(det>1e-15 and ratio < boundary and not np.isclose(ratio,boundary,rtol=1e-12,atol=1e-12)),trace,det,ratio


def filter_keypoints(candidates: list[Keypoint], pyramid: tuple[Octave,...],
                     contrast: float = .008, edge_ratio: float = 10.0) -> list[Keypoint]:
    """Raw |D_hat| in [0,1] image units, no division by intervals; one status each."""
    if not np.isfinite(contrast) or contrast < 0:
        raise ValueError("contrast must be nonnegative")
    result = []
    for candidate in candidates:
        kp = refine_keypoint(candidate,pyramid[candidate.octave])
        if kp.status == "invalid":
            result.append(kp); continue
        if abs(kp.response) < contrast:
            result.append(replace(kp,status="low contrast",reason=f"|D̂|={abs(kp.response):.5f} < {contrast:.5f}")); continue
        hxy = calculate_spatial_hessian(pyramid[kp.octave].dogs,kp.layer,kp.row,kp.col)
        keep,tr,det,ratio = edge_decision(hxy,edge_ratio)
        if not keep:
            result.append(replace(kp,status="edge",reason=f"det={det:.5g}; trace²/det={ratio:.3g}; ngưỡng={(edge_ratio+1)**2/edge_ratio:.3g}")); continue
        result.append(replace(kp,status="kept",reason=f"|D̂|={abs(kp.response):.5f}; curvature={ratio:.3f}"))
    return result


def gaussian_level(kp: Keypoint, octave: Octave) -> tuple[np.ndarray,int]:
    """Nearest available Gaussian level to fractional DoG index (no scale interpolation)."""
    j = int(np.clip(np.rint(kp.layer_local),0,len(octave.gaussians)-1))
    return octave.gaussians[j],j


def gradient_field(image: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Central differences with REFLECT_101 boundary; theta uses image y-down."""
    a = np.asarray(image,dtype=float)
    k = np.array([[-.5,0,.5]],dtype=float)
    return (cv2.filter2D(a,-1,k,borderType=cv2.BORDER_REFLECT_101),
            cv2.filter2D(a,-1,k.T,borderType=cv2.BORDER_REFLECT_101))


def calculate_orientation_histogram(kp: Keypoint, octave: Octave) -> Orientation:
    image,j = gaussian_level(kp,octave)
    gx,gy = gradient_field(image)
    radius = int(round(4.5*kp.sigma_local))
    weight_sigma = 1.5*kp.sigma_local
    hist = np.zeros(36,dtype=float)
    contributions = []
    for row in range(max(1,int(kp.row_local)-radius),min(image.shape[0]-1,int(kp.row_local)+radius+2)):
        for col in range(max(1,int(kp.col_local)-radius),min(image.shape[1]-1,int(kp.col_local)+radius+2)):
            dx,dy = col-kp.col_local,row-kp.row_local
            if dx*dx+dy*dy > radius*radius: continue
            mag = float(np.hypot(gx[row,col],gy[row,col]))
            angle = float(np.degrees(np.arctan2(gy[row,col],gx[row,col]))%360)
            spatial = float(np.exp(-(dx*dx+dy*dy)/(2*weight_sigma**2)))
            weight = mag*spatial
            u = angle/10; low = int(np.floor(u))%36; frac = u-np.floor(u)
            hist[low] += weight*(1-frac); hist[(low+1)%36] += weight*frac
            contributions.append((row,col,mag,angle,spatial,weight))
    smooth = (np.roll(hist,2)+4*np.roll(hist,1)+6*hist+4*np.roll(hist,-1)+np.roll(hist,-2))/16
    maximum = float(smooth.max())
    indices = [i for i,v in enumerate(smooth) if maximum>0 and v>=.8*maximum
               and v>smooth[(i-1)%36] and v>=smooth[(i+1)%36]]
    indices.sort(key=lambda i:(-smooth[i],i))
    peaks = tuple(float(10*i) for i in indices)
    return Orientation(smooth,hist,tuple(contributions),peaks,j)


def assign_keypoint_orientations(kp: Keypoint, octave: Octave) -> tuple[Keypoint,...]:
    info = calculate_orientation_histogram(kp,octave)
    return tuple(replace(kp,orientation=angle) for angle in info.peaks)


def normalize_sift_descriptor(raw: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    d = np.asarray(raw,dtype=float).reshape(-1)
    if d.shape != (128,) or not np.isfinite(d).all() or np.any(d<0):
        raise ValueError("Expected finite nonnegative 128D histogram")
    norm = np.linalg.norm(d)
    first = d/norm if norm>0 else np.zeros_like(d)
    clipped = np.minimum(first,.2)
    norm2 = np.linalg.norm(clipped)
    final = clipped/norm2 if norm2>0 else np.zeros_like(d)
    return first,clipped,final


def descriptor_index(cell_row: int, cell_col: int, orientation_bin: int) -> int:
    if not 0<=cell_row<4 or not 0<=cell_col<4 or not 0<=orientation_bin<8:
        raise ValueError("Invalid descriptor index")
    return (cell_row*4+cell_col)*8+orientation_bin


def build_sift_descriptor(kp: Keypoint, octave: Octave) -> Descriptor:
    """4x4x8 trilinear histogram; cell width=3 sigma_local; skip out-of-image samples."""
    if kp.orientation is None:
        raise ValueError("Orientation is required")
    image,_ = gaussian_level(kp,octave)
    gx,gy = gradient_field(image)
    theta = radians(kp.orientation); ct,st = cos(theta),sin(theta)
    width = 3*kp.sigma_local
    radius = int(ceil(width*2.5*sqrt(2)))
    cells = np.zeros((4,4,8),dtype=float)
    samples = []
    for row in range(max(1,int(kp.row_local)-radius),min(image.shape[0]-1,int(kp.row_local)+radius+2)):
        for col in range(max(1,int(kp.col_local)-radius),min(image.shape[1]-1,int(kp.col_local)+radius+2)):
            dx,dy = col-kp.col_local,row-kp.row_local
            u = (ct*dx+st*dy)/width+1.5
            v = (-st*dx+ct*dy)/width+1.5
            if not -.5<u<3.5 or not -.5<v<3.5: continue
            mag = float(np.hypot(gx[row,col],gy[row,col]))
            angle = float((np.degrees(np.arctan2(gy[row,col],gx[row,col]))-kp.orientation)%360)
            weight = mag*np.exp(-(dx*dx+dy*dy)/(2*(2*width)**2))
            b = angle/45
            u0,v0,b0 = int(np.floor(u)),int(np.floor(v)),int(np.floor(b))
            for rr in (v0,v0+1):
                if not 0<=rr<4: continue
                wy = 1-abs(v-rr)
                for cc in (u0,u0+1):
                    if not 0<=cc<4: continue
                    wx = 1-abs(u-cc)
                    for bb in (b0,b0+1):
                        wb = 1-abs(b-bb)
                        cells[rr,cc,bb%8] += weight*wy*wx*wb
            samples.append((row,col,u,v,angle,float(weight)))
    raw = cells.reshape(128).copy()
    first,clipped,final = normalize_sift_descriptor(raw)
    return Descriptor(raw,first,clipped,final,cells,tuple(samples))


def sample_oriented_descriptor_patch(kp: Keypoint, octave: Octave, size: int = 48) -> tuple[np.ndarray,np.ndarray]:
    """True bilinear sampling in descriptor frame; mask identifies out-of-image sites."""
    if kp.orientation is None or size<16:
        raise ValueError("Need oriented keypoint and size >=16")
    image,_ = gaussian_level(kp,octave)
    width = 3*kp.sigma_local
    axis = (np.arange(size,dtype=float)-(size-1)/2)*(4*width/size)
    vv,uu = np.meshgrid(axis,axis,indexing="ij")
    theta = radians(kp.orientation); ct,st = cos(theta),sin(theta)
    xx = kp.col_local+ct*uu-st*vv
    yy = kp.row_local+st*uu+ct*vv
    valid = (xx>=0)&(xx<=image.shape[1]-1)&(yy>=0)&(yy<=image.shape[0]-1)
    patch = cv2.remap(image,xx.astype(np.float32),yy.astype(np.float32),
                      cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    patch[~valid]=0
    return patch,valid


def describe(pyramid: tuple[Octave,...], points: list[Keypoint], limit: int = 80) -> list[Keypoint]:
    result = []
    for kp in sorted((p for p in points if p.status=="kept"),key=lambda p:-abs(p.response))[:limit]:
        for oriented in assign_keypoint_orientations(kp,pyramid[kp.octave]):
            descriptor = build_sift_descriptor(oriented,pyramid[oriented.octave])
            if np.linalg.norm(descriptor.final)>0:
                result.append(replace(oriented,descriptor=descriptor.final))
    return result


def pipeline(image: np.ndarray, *, octaves: int = 3, intervals: int = 3,
             base_sigma: float = 1.6, contrast: float = .008, edge_ratio: float = 10.0,
             with_descriptors: bool = True) -> tuple[tuple[Octave,...],list[Keypoint],list[Keypoint],list[Keypoint]]:
    pyramid = build_pyramid(image,octaves,intervals,base_sigma)
    raw = detect_dog_candidates(pyramid)
    filtered = filter_keypoints(raw,pyramid,contrast,edge_ratio)
    described = describe(pyramid,filtered) if with_descriptors else []
    return pyramid,raw,filtered,described


def compute_descriptor_distances(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    q = np.asarray(query,dtype=float).reshape(128)
    t = np.asarray(targets,dtype=float).reshape(-1,128)
    return np.linalg.norm(t-q,axis=1)


def apply_lowe_ratio_test(distances: np.ndarray, threshold: float = .8, query: int = 0) -> Match:
    if not 0<threshold<=1:
        raise ValueError("Ratio threshold must be in (0,1]")
    d = np.asarray(distances,dtype=float).reshape(-1)
    if len(d)<2 or not np.isfinite(d).all() or np.any(d<0):
        return Match(query,None,None,None,None,None,False,"Cần ít nhất hai descriptor đích hợp lệ")
    order = np.argsort(d,kind="stable"); i,j = int(order[0]),int(order[1])
    d1,d2 = float(d[i]),float(d[j])
    if d2==0:
        return Match(query,i,j,d1,d2,None,False,"Khoảng cách thứ hai bằng 0: trùng/không phân biệt")
    ratio = d1/d2
    accept = bool(ratio<threshold)
    return Match(query,i,j,d1,d2,ratio,accept,
                 "Phân biệt đủ theo ratio test" if accept else "Hai láng giềng quá giống nhau")


def match_descriptors(a: list[Keypoint], b: list[Keypoint], threshold: float = .8) -> list[Match]:
    target = np.array([p.descriptor for p in b if p.descriptor is not None],dtype=float).reshape(-1,128)
    return [apply_lowe_ratio_test(compute_descriptor_distances(p.descriptor,target),threshold,i)
            for i,p in enumerate(a) if p.descriptor is not None]


def transform_scene(image: np.ndarray, rotation: float = 20, scale: float = 1.0,
                    brightness: float = 0, contrast: float = 1, noise: float = 0,
                    seed: int = 2026) -> tuple[np.ndarray,np.ndarray]:
    """Known forward source->destination affine; truth is evaluation only."""
    a = np.asarray(image,dtype=float)
    matrix = cv2.getRotationMatrix2D(((a.shape[1]-1)/2,(a.shape[0]-1)/2),rotation,scale)
    transformed = cv2.warpAffine(a,matrix,(a.shape[1],a.shape[0]),
                                 flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    rng = np.random.default_rng(seed)
    transformed = np.clip(contrast*transformed+brightness+rng.normal(0,noise,a.shape),0,1)
    return transformed,matrix


def evaluate_geometric_correspondence(a: Keypoint, b: Keypoint, matrix: np.ndarray,
                                      tolerance: float = 4.0) -> tuple[float,bool]:
    xy = np.asarray(matrix,dtype=float)@np.array([a.col_global,a.row_global,1.])
    error = float(np.hypot(xy[0]-b.col_global,xy[1]-b.row_global))
    return error,bool(error<=tolerance)
