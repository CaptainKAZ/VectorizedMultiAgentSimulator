#  Copyright (c) 2026.
#  Vectorised 2D geometry for the RM 3v3 environment.
#
#  All functions operate on torch tensors and are written to be readable first,
#  fast second.  The training-side optimiser can replace the hot paths later
#  without changing the rule semantics.
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch

Tensor = torch.Tensor

PI = math.pi
TWO_PI = 2.0 * math.pi
EPS = 1e-9


# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------
def wrap_angle(a: Tensor) -> Tensor:
    """Wrap angle(s) to [-pi, pi)."""
    return (a + PI) % TWO_PI - PI


def rotate(vec: Tensor, ang: Tensor) -> Tensor:
    """Rotate 2D vector(s) by angle(s).

    vec: (..., 2), ang: (...) -> (..., 2)
    """
    c = torch.cos(ang)
    s = torch.sin(ang)
    x = vec[..., 0]
    y = vec[..., 1]
    return torch.stack((x * c - y * s, x * s + y * c), dim=-1)


def polygon_tensor(poly: Sequence[Sequence[float]], device, dtype=torch.float32) -> Tensor:
    return torch.tensor(poly, device=device, dtype=dtype)


def polygon_signed_area(poly: Tensor) -> Tensor:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.sum(x * torch.roll(y, -1) - torch.roll(x, -1) * y)


def ensure_ccw(poly: Tensor) -> Tensor:
    if polygon_signed_area(poly) < 0:
        return torch.flip(poly, dims=(0,))
    return poly


# ---------------------------------------------------------------------------
# point / segment / polygon
# ---------------------------------------------------------------------------
def point_in_polygon(points: Tensor, poly: Tensor) -> Tensor:
    """Ray-casting point-in-polygon.

    points: (B, 2), poly: (P, 2) -> (B,) bool.
    Boundary points may be reported either way; callers that care about the
    boundary also run segment intersection tests.
    """
    if points.numel() == 0:
        return torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)
    x = points[..., 0]
    y = points[..., 1]
    inside = torch.zeros_like(x, dtype=torch.bool)
    n = poly.shape[0]
    for k in range(n):
        x1, y1 = poly[k, 0], poly[k, 1]
        x2, y2 = poly[(k + 1) % n, 0], poly[(k + 1) % n, 1]
        crosses = (y1 > y) != (y2 > y)
        # avoid division by zero for horizontal edges (crosses is False there)
        denom = torch.where(torch.abs(y2 - y1) < EPS, torch.full_like(y2 - y1, EPS), y2 - y1)
        x_int = (x2 - x1) * (y - y1) / denom + x1
        inside = inside ^ (crosses & (x < x_int))
    return inside


def closest_point_on_segment(points: Tensor, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Closest point on segment a-b for each point.

    points: (B, 2), a/b: (2,) or (B, 2) -> q: (B, 2), t: (B,)
    """
    a = a.expand_as(points) if a.ndim == 1 else a
    b = b.expand_as(points) if b.ndim == 1 else b
    ab = b - a
    denom = torch.sum(ab * ab, dim=-1).clamp_min(EPS)
    t = torch.sum((points - a) * ab, dim=-1) / denom
    t = t.clamp(0.0, 1.0)
    q = a + t.unsqueeze(-1) * ab
    return q, t


def nearest_point_on_polygon(
    points: Tensor, poly: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Closest point on a solid polygon boundary.

    Returns (q, dist, normal, inside):
      q:      (B, 2) closest point on the boundary
      dist:   (B,)   distance from point to boundary
      normal: (B, 2) unit normal; outside -> from q to point, inside -> edge outward
      inside: (B,) bool
    """
    B = points.shape[0]
    device = points.device
    inside = point_in_polygon(points, poly)

    n = poly.shape[0]
    best_d = torch.full((B,), float("inf"), device=device)
    best_q = torch.zeros((B, 2), device=device)
    best_n = torch.zeros((B, 2), device=device)

    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        q, _ = closest_point_on_segment(points, a, b)
        d = torch.linalg.vector_norm(points - q, dim=-1)
        # outward normal for CCW polygon: rotate edge direction by -90 deg
        e = b - a
        en = torch.linalg.vector_norm(e).clamp_min(EPS)
        outward = torch.stack((e[1], -e[0]), dim=-1) / en
        outside_n = points - q
        on = torch.linalg.vector_norm(outside_n, dim=-1).clamp_min(EPS)
        normal = torch.where(
            inside.unsqueeze(-1),
            outward.unsqueeze(0).expand_as(outside_n),
            outside_n / on.unsqueeze(-1),
        )
        better = d < best_d
        best_d = torch.where(better, d, best_d)
        best_q = torch.where(better.unsqueeze(-1), q, best_q)
        best_n = torch.where(better.unsqueeze(-1), normal, best_n)

    # A point inside the polygon has dist 0 by definition; keep the distance to
    # the nearest edge as the penetration depth in that case.
    return best_q, best_d, best_n, inside


# ---------------------------------------------------------------------------
# segment intersections (closed set: touching counts as blocked)
# ---------------------------------------------------------------------------
def _orient(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (c[..., 0] - a[..., 0])


def _on_segment(a: Tensor, b: Tensor, p: Tensor) -> Tensor:
    return (
        (p[..., 0] >= torch.minimum(a[..., 0], b[..., 0]) - EPS)
        & (p[..., 0] <= torch.maximum(a[..., 0], b[..., 0]) + EPS)
        & (p[..., 1] >= torch.minimum(a[..., 1], b[..., 1]) - EPS)
        & (p[..., 1] <= torch.maximum(a[..., 1], b[..., 1]) + EPS)
    )


def segments_intersect(p1: Tensor, p2: Tensor, q1: Tensor, q2: Tensor) -> Tensor:
    """Closed-set segment intersection, vectorised over the leading dims."""
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)

    proper = (o1 * o2 < 0) & (o3 * o4 < 0)
    touch = (
        ((o1.abs() <= EPS) & _on_segment(p1, p2, q1))
        | ((o2.abs() <= EPS) & _on_segment(p1, p2, q2))
        | ((o3.abs() <= EPS) & _on_segment(q1, q2, p1))
        | ((o4.abs() <= EPS) & _on_segment(q1, q2, p2))
    )
    return proper | touch


def segment_intersects_circle(
    p1: Tensor, p2: Tensor, center: Tensor, radius: Tensor
) -> Tensor:
    """Closed-set segment vs circle."""
    q, _ = closest_point_on_segment(center, p1, p2)
    d = torch.linalg.vector_norm(center - q, dim=-1)
    return d <= radius + EPS


def segment_intersects_polygon(p1: Tensor, p2: Tensor, poly: Tensor) -> Tensor:
    """Closed-set segment vs solid polygon."""
    blocked = point_in_polygon(p1, poly) | point_in_polygon(p2, poly)
    n = poly.shape[0]
    for k in range(n):
        q1 = poly[k].expand_as(p1)
        q2 = poly[(k + 1) % n].expand_as(p1)
        blocked = blocked | segments_intersect(p1, p2, q1, q2)
    return blocked


# ---------------------------------------------------------------------------
# ray casting
# ---------------------------------------------------------------------------
def ray_polygon(
    origin: Tensor, direction: Tensor, poly: Tensor, check_inside: bool = True
) -> Tensor:
    """First positive ray/solid-polygon hit distance.

    origin/direction: (..., 2) -> t: (...,) with inf for no hit.
    If check_inside is False, the caller guarantees the origin is outside the
    polygon; this skips a costly point-in-polygon test (used by visibility).
    """
    shape = origin.shape[:-1]
    origin, direction = torch.broadcast_tensors(origin, direction)
    shape = origin.shape[:-1]
    o = origin.reshape(-1, 2)
    d = direction.reshape(-1, 2)
    t_best = torch.full((o.shape[0],), float("inf"), device=o.device)

    if check_inside:
        inside = point_in_polygon(o, poly)
        t_best = torch.where(inside, torch.zeros_like(t_best), t_best)

    n = poly.shape[0]
    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        s = b - a
        denom = d[:, 0] * s[1] - d[:, 1] * s[0]
        valid = torch.abs(denom) > EPS
        q = a.unsqueeze(0) - o
        t = (q[:, 0] * s[1] - q[:, 1] * s[0]) / torch.where(valid, denom, torch.ones_like(denom))
        u = (q[:, 0] * d[:, 1] - q[:, 1] * d[:, 0]) / torch.where(valid, denom, torch.ones_like(denom))
        hit = valid & (t >= 0.0) & (u >= -EPS) & (u <= 1.0 + EPS)
        t_best = torch.where(hit & (t < t_best), t, t_best)
    return t_best.reshape(shape)


def ray_circle(origin: Tensor, direction: Tensor, center: Tensor, radius: Tensor) -> Tensor:
    """First positive ray/circle hit distance."""
    shape = origin.shape[:-1]
    origin, direction = torch.broadcast_tensors(origin, direction)
    shape = origin.shape[:-1]
    o = origin.reshape(-1, 2)
    d = direction.reshape(-1, 2)
    c = center.reshape(-1, 2)
    r = radius.reshape(-1)
    oc = o - c
    b = torch.sum(oc * d, dim=-1)
    c0 = torch.sum(oc * oc, dim=-1) - r * r
    disc = b * b - c0
    valid = disc >= 0
    sq = torch.sqrt(disc.clamp_min(0))
    t0 = -b - sq
    t1 = -b + sq
    t = torch.where(t0 >= 0, t0, t1)
    t = torch.where(valid & (t >= 0), t, torch.full_like(t, float("inf")))
    return t.reshape(shape)


def ray_wall(origin: Tensor, direction: Tensor, hx: float, hy: float) -> Tensor:
    """Distance to the first axis-aligned wall hit (origin assumed inside)."""
    ox = origin[..., 0]
    oy = origin[..., 1]
    dx = direction[..., 0]
    dy = direction[..., 1]
    ts = []
    for val, o, d, sign in (
        (-hx, ox, dx, -1.0),
        (hx, ox, dx, 1.0),
        (-hy, oy, dy, -1.0),
        (hy, oy, dy, 1.0),
    ):
        t = (val - o) / torch.where(torch.abs(d) < EPS, torch.full_like(d, EPS), d)
        ts.append(torch.where(t > 1e-6, t, torch.full_like(t, float("inf"))))
    return torch.stack(ts, dim=-1).min(dim=-1).values


def ray_edges(origin: Tensor, direction: Tensor, edges: Tensor) -> Tuple[Tensor, Tensor]:
    """Ray vs a batch of segments, fully vectorised.

    origin/direction: (..., 2)
    edges: (E, 4) as x1,y1,x2,y2
    returns t (..., E) and hit (..., E) bool.
    """
    origin, direction = torch.broadcast_tensors(origin, direction)
    o = origin.unsqueeze(-2)
    d = direction.unsqueeze(-2)
    q1 = edges[:, :2]
    q2 = edges[:, 2:]
    s = q2 - q1                                   # (E,2)
    denom = d[..., 0] * s[:, 1] - d[..., 1] * s[:, 0]
    valid = torch.abs(denom) > EPS
    q = q1 - o                                    # (...,E,2)
    denom_safe = torch.where(valid, denom, torch.ones_like(denom))
    t = (q[..., 0] * s[:, 1] - q[..., 1] * s[:, 0]) / denom_safe
    u = (q[..., 0] * d[..., 1] - q[..., 1] * d[..., 0]) / denom_safe
    hit = valid & (t >= 0.0) & (u >= -EPS) & (u <= 1.0 + EPS)
    return t, hit


def segment_circle_blocked(
    observer_pos: Tensor, points: Tensor, circles: Tensor
) -> Tensor:
    """Whether segment observer_pos -> each point is blocked by any circle.

    observer_pos: (B,2), points: (B,S,2), circles: (B,K,3) -> (B,S) bool.
    """
    if circles.numel() == 0:
        return torch.zeros(points.shape[:-1], device=points.device, dtype=torch.bool)
    c = circles[:, None, :, :2]                    # (B,1,K,2)
    r = circles[:, None, :, 2]                     # (B,1,K)
    ab = points[:, :, None, :] - observer_pos[:, None, None, :]  # (B,S,1,2)
    denom = torch.sum(ab * ab, dim=-1).clamp_min(EPS)            # (B,S,1)
    t = torch.sum((c - observer_pos[:, None, None, :]) * ab, dim=-1) / denom
    t = t.clamp(0.0, 1.0)
    q = observer_pos[:, None, None, :] + t.unsqueeze(-1) * ab
    dist = torch.linalg.vector_norm(c - q, dim=-1)               # (B,S,K)
    return (dist <= r + EPS).any(dim=-1)



# ---------------------------------------------------------------------------
# visibility arc
# ---------------------------------------------------------------------------
def visibility_arc(
    observer_pos: Tensor,          # (B, 2)
    observer_yaw: Tensor,          # (B,)
    target_pos: Tensor,            # (B, 2)
    target_radius: float,
    occluder_circles: Tensor,      # (B, K, 3)  x, y, r
    occluder_edges: Tensor,        # (E, 4) x1,y1,x2,y2
    n_samples: int,
) -> Tensor:
    """Visible fraction of the target's front arc.

    Vectorised over batch, samples, circles and all cover edges.  The cover
    polygons are represented as a flat edge list; a sample is blocked if the
    segment O->P hits any edge with 0 <= t < 1.
    """
    B = observer_pos.shape[0]
    device = observer_pos.device
    rel = target_pos - observer_pos
    d = torch.linalg.vector_norm(rel, dim=-1)
    d_safe = torch.clamp(d, min=2.0 * target_radius)
    ratio = torch.clamp(
        torch.full_like(d_safe, target_radius) / d_safe, -1.0, 1.0
    )
    full = 2.0 * torch.acos(ratio)

    u = torch.where(
        (d > 1e-6).unsqueeze(-1),
        -rel / d.clamp_min(1e-6).unsqueeze(-1),
        torch.tensor([1.0, 0.0], device=device, dtype=observer_pos.dtype).expand(B, 2),
    )

    S = n_samples
    phi = -full.unsqueeze(-1) / 2.0 + (
        torch.arange(S, device=device, dtype=observer_pos.dtype).unsqueeze(0) + 0.5
    ) * full.unsqueeze(-1) / float(S)                       # (B, S)
    P = target_pos.unsqueeze(1) + target_radius * rotate(u.unsqueeze(1), phi)  # (B, S, 2)

    blocked = segment_circle_blocked(observer_pos, P, occluder_circles)
    if occluder_edges.numel() > 0:
        seg = P - observer_pos.unsqueeze(1)                 # (B,S,2)
        t, hit = ray_edges(observer_pos.unsqueeze(1), seg, occluder_edges)
        blocked = blocked | (hit & (t < 1.0 - 1e-6)).any(dim=-1)

    return (~blocked).to(observer_pos.dtype).mean(dim=1)


# ---------------------------------------------------------------------------
# lidar
# ---------------------------------------------------------------------------
def lidar_ranges(
    origin: Tensor,                # (B, 2)
    yaw: Tensor,                   # (B,)
    edges: Tensor,                 # (E, 4)
    circles: Tensor,               # (B, K, 3)
    n_rays: int,
    max_range: float,
    fov_half: float,
    hx: float,
    hy: float,
) -> Tensor:
    """Normalised 128-ray lidar in the agent's front 180°.

    Returns (B, n_rays) in [0, 1]; no hit is 1.0.
    """
    B = origin.shape[0]
    device = origin.device
    angles = yaw.unsqueeze(-1) + torch.linspace(
        -fov_half, fov_half, n_rays, device=device, dtype=origin.dtype
    ).unsqueeze(0)
    dirs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)  # (B,R,2)

    o = origin.unsqueeze(1).expand(B, n_rays, 2)
    best = torch.full((B, n_rays), float("inf"), device=device)

    if edges.numel() > 0:
        t, hit = ray_edges(o, dirs, edges)
        t = torch.where(hit, t, torch.full_like(t, float("inf")))
        best = torch.minimum(best, t.min(dim=-1).values)

    if circles.numel() > 0:
        c = circles[:, None, :, :2]                    # (B,1,K,2)
        r = circles[:, None, :, 2]                     # (B,1,K)
        oc = o[:, :, None, :] - c                      # (B,R,K,2)
        b = torch.sum(oc * dirs[:, :, None, :], dim=-1)
        c0 = torch.sum(oc * oc, dim=-1) - r * r
        disc = b * b - c0
        valid = disc >= 0
        sq = torch.sqrt(disc.clamp_min(0))
        t0 = -b - sq
        t1 = -b + sq
        t = torch.where(t0 >= 0, t0, t1)
        t = torch.where(valid & (t >= 0), t, torch.full_like(t, float("inf")))
        best = torch.minimum(best, t.min(dim=-1).values)

    t_wall = ray_wall(o, dirs, hx, hy)
    best = torch.minimum(best, t_wall)

    best = torch.clamp(best, max=max_range)
    out = best / max_range
    out = torch.where(torch.isinf(best), torch.ones_like(out), out)
    return out
