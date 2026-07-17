"""
SVG Logo Reward Function v2 — 课程作业 Part B（增强版）

评估维度（4 维 + Sonnet Judge）:
  1. 结构有效性   — XML 合法性、标签平衡、截断检测
  2. 设计合规性   — 禁止元素、标签白名单、颜色使用
  3. 视觉质量     — 坐标范围、元素居中、可见性
  4. 提示词对齐度 — 颜色匹配、形状检测、语义分组、复杂度

改进点（v1 → v2）:
  - 颜色名映射 40→120+，RGB欧氏距离→CIE76感知距离
  - ElementTree 树遍历替代纯正则形状检测，大幅减少误报
  - 新增截断检测、标签平衡、失控生成长度检测
  - 语义关键词分组（nature/geometry/food/abstract…）
  - 阶梯式对齐评分，消除 GT 分数顶部聚集
  - Sonnet Judge 缓存+重试+结构化评判
  - 失败原因列表

返回:
  total: 0.0 ~ 1.0
  detail: 各维度得分 + failure_reasons
"""

from __future__ import annotations

import re
import json
import math
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Set

# ============================================================================
# 权重（新增 visual 维度）
# ============================================================================
WEIGHTS = {
    "structural": 0.25,    # ↓0.05
    "compliance": 0.20,    # ↓0.05
    "visual":     0.15,    # NEW
    "alignment":  0.40,    # ↓0.05
}

# ============================================================================
# 禁止规则
# ============================================================================
BANNED_ELEMENTS = {"image", "script", "foreignObject", "use", "iframe", "style"}
BANNED_PATTERNS = [
    r"href\s*=\s*[\"\']", r"xlink:href\s*=", r"javascript\s*:",
    r"<script", r"on\w+\s*=",
]
ALLOWED_TAGS = {
    "svg", "defs", "g", "path", "circle", "ellipse",
    "rect", "line", "polygon", "polyline",
    "linearGradient", "radialGradient", "stop",
    "clipPath", "filter", "feGaussianBlur", "feMerge", "feMergeNode",
    "mask", "pattern", "title", "desc", "metadata",
}
DRAWING_TAGS = {"circle", "ellipse", "rect", "line", "polygon", "polyline", "path"}

# ============================================================================
# 颜色名 → hex 映射（120+ 项，覆盖 prompt 中常见的颜色表述）
# ============================================================================
COLOR_NAME_TO_HEX = {
    # 基础色
    "white": "#ffffff", "black": "#1b1b1e", "cream": "#fbf3e3",
    "off-white": "#fff8f0", "ivory": "#fffff0", "snow": "#fffafa",
    "warm cream": "#fbf3e3", "pale cream": "#fef9f0",
    # 蓝系
    "navy": "#1b3a5c", "deep navy": "#0a1730", "dark navy": "#101b33",
    "blue": "#2a6fdb", "deep blue": "#1f4e79", "sky blue": "#8ecae6",
    "pale sky blue": "#bfe8f5", "slate blue": "#2a4759",
    "medium blue": "#5b8fb9", "teal": "#2ec4b6", "deep teal": "#0f5c56",
    "teal-blue": "#0f5e66", "deep teal-blue": "#0a4a52",
    "cyan": "#22e5ff", "light teal": "#5fc7c2",
    # 绿系
    "green": "#4caf50", "forest green": "#1f4d33", "deep forest green": "#173d24",
    "sage green": "#cddfc4", "sage": "#cddfc4", "pale sage-green": "#cddfc4",
    "muted sage green": "#7f9a6e", "olive": "#808000", "olive-green": "#8f9a3d",
    "leafy green": "#4caf7d", "mint": "#98fb98", "deep green": "#2f6b4f",
    "dark forest-green": "#0e2418",
    # 橙/红系
    "orange": "#ff6b35", "warm orange": "#f2994a", "deep orange": "#e08324",
    "burnt orange": "#b85a18", "mustard orange": "#c9922f",
    "coral": "#ff6f5e", "coral red": "#ff6f5e",
    "red": "#d64541", "burgundy": "#8f2438", "deep red": "#5e1424",
    "mango": "#ffc145", "mango yellow": "#ffc145",
    "peach": "#ffdab9", "salmon": "#fa8072", "tomato": "#ff6347",
    # 黄/金系
    "yellow": "#ffd93b", "gold": "#f2a93b", "golden": "#d4a017",
    "mustard": "#d2a531", "golden mustard": "#d2a531",
    "warm gold": "#f0cf7a", "golden-tan": "#e5b76a",
    "bright yellow-green": "#d4ea3a",
    "mustard yellow": "#d2a531", "muted gold": "#c9a24b",
    "soft golden": "#f4d98a", "deep amber": "#b85a18", "amber": "#f2a93b",
    # 棕/大地系
    "brown": "#5c4326", "dark brown": "#3b2410", "deep brown": "#2b1f18",
    "walnut brown": "#6b4226", "sandy beige": "#e7d9b0",
    "charcoal": "#2d2a26", "dark charcoal": "#2b2b2b",
    "deep charcoal": "#2d2a26", "charcoal-brown": "#3a2b20",
    "slate": "#2a4759", "taupe": "#8b7355", "copper": "#b87333",
    # 紫/粉系
    "purple": "#7c3aed", "violet": "#a239ff", "lavender": "#c4b5fd",
    "pink": "#ff9eb5", "magenta": "#ff00ff", "mauve": "#e0b0ff",
    "bright violet": "#a239ff",
    # 灰系
    "gray": "#888888", "grey": "#888888", "dark gray": "#444444",
    "light gray": "#cccccc", "silver": "#c0c0c0",
    "pale gray-blue": "#e8ecef", "warm gray": "#a08c7a",
    # 特殊
    "transparent": "none", "maroon": "#800000", "indigo": "#6366f1",
    "khaki": "#c3b091", "tan": "#d2b48c", "beige": "#f5f5dc",
    "rose": "#ff007f", "aqua": "#00ffff", "lime": "#00ff00",
}
COLOR_NAMES = sorted(COLOR_NAME_TO_HEX.keys(), key=len, reverse=True)

# ============================================================================
# 语义关键词分组
# ============================================================================
SEMANTIC_GROUPS = {
    "nature": {
        "keywords": {"leaf", "leaves", "tree", "sprout", "stem", "flower", "petal",
                      "sun", "vine", "branch", "plant", "forest", "pine", "seed", "bud",
                      "bloom", "blossom", "botanical", "root", "grass", "herb", "spice"},
        "svg_indicators": ["leaf", "tree", "flower", "sun", "sprout", "petal"],
    },
    "geometry": {
        "keywords": {"circle", "hexagon", "triangle", "square", "rectangle", "ring",
                      "octagon", "diamond", "badge", "shield", "frame", "border", "outline",
                      "emblem", "seal", "medallion", "coin"},
        "svg_indicators": ["circle", "hexagon", "triangle", "rectangle", "badge",
                           "shield", "ring", "arch"],
    },
    "food_drink": {
        "keywords": {"bottle", "cup", "mug", "plate", "fork", "spoon", "knife",
                      "glass", "bowl", "jar", "dish", "drink", "beer", "wine", "coffee",
                      "tea", "brew", "smoothie", "juice", "fruit", "berry", "nut",
                      "bean", "malt", "hop", "grain", "bake", "bakery"},
        "svg_indicators": ["bottle", "cup", "mug", "fork", "droplet"],
    },
    "art_craft": {
        "keywords": {"brush", "paint", "pen", "pencil", "crayon", "palette",
                      "canvas", "draw", "sketch", "doodle", "ink", "craft", "art",
                      "handmade", "artisanal", "design", "studio", "creative", "sparkle"},
        "svg_indicators": ["brush", "paint", "sparkle", "swirl", "ribbon"],
    },
    "music_sound": {
        "keywords": {"note", "music", "sound", "melody", "rhythm", "song", "tune",
                      "audio", "podcast", "wave", "signal", "frequency", "treble", "bass"},
        "svg_indicators": ["note", "wave", "swoosh"],
    },
    "health_wellness": {
        "keywords": {"spine", "hand", "body", "health", "wellness", "care", "heal",
                      "therapy", "support", "comfort", "nurture", "growth", "mind",
                      "mental", "organic", "natural", "pure", "fresh", "clean"},
        "svg_indicators": ["spine", "hand", "leaf", "flower", "sun"],
    },
    "tech_digital": {
        "keywords": {"play", "controller", "game", "gaming", "video", "screen",
                      "digital", "tech", "code", "data", "network", "connect", "signal",
                      "button", "icon", "app", "software", "platform", "startup"},
        "svg_indicators": ["arrow", "shield", "badge", "circle", "triangle"],
    },
    "home_shelter": {
        "keywords": {"house", "home", "roof", "shelter", "building", "door",
                      "window", "room", "space", "place", "residence", "dwelling",
                      "foundation", "structure", "architecture"},
        "svg_indicators": ["house", "roof", "arch", "column"],
    },
}


# ============================================================================
# SVG 视觉结构检测模式（精确版 + 树遍历版）
# ============================================================================
SHAPE_REGEX = {
    # 仅对 path d 属性做精细模式匹配
    "swirl":    r'd\s*=\s*"[^"]*[cC]\s*-?\d[^"]*[cC]\s*-?\d[^"]*[cC]',
    "ribbon":   r'd\s*=\s*"[^"]*(?:[cC]\s*-?\d[^"]*){3,}',
    "wave":     r'd\s*=\s*"[^"]*(?:[qQ]\s*-?\d[^"]*){2,}',
    "swoosh":   r'd\s*=\s*"[^"]*(?:[qQcC]\s*-?\d[^"]*){2,}',
    "arch":     r'd\s*=\s*"[^"]*(?:[Aa]\s*-?\d[^"]*){1,}',
    "complex_path": r'd\s*=\s*"[^"]{80,}"',  # 复杂路径 ≥80 字符
    "sparkle":  r'd\s*=\s*"[^"]*(?:L\s*-?\d[^"]*){4,}"',
    "star_like": r'points\s*=\s*"[^"]{30,}"',  # 多点 polygon
}


def _parse_svg_tree(svg: str) -> Tuple[Optional[ET.Element], Dict[str, Any]]:
    """解析 SVG 并收集树结构信息."""
    info: Dict[str, Any] = {"tag_counts": Counter(), "has_defs": False,
                             "has_gradient": False, "element_count": 0,
                             "drawing_elements": 0, "max_depth": 0,
                             "ids": [], "coords_out_of_bounds": 0,
                             "total_drawing_elements": 0}
    try:
        cleaned = svg.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        root = ET.fromstring(cleaned)
    except ET.ParseError:
        return None, info

    tag_ns = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag_ns != "svg":
        return root, info

    # BFS 遍历
    stack = [(root, 0)]
    while stack:
        elem, depth = stack.pop()
        info["max_depth"] = max(info["max_depth"], depth)
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        info["tag_counts"][tag] += 1
        info["element_count"] += 1

        if tag in DRAWING_TAGS:
            info["total_drawing_elements"] += 1

        if tag == "defs":
            info["has_defs"] = True
        if tag in ("linearGradient", "radialGradient"):
            info["has_gradient"] = True

        elem_id = elem.get("id", "")
        if elem_id:
            info["ids"].append(elem_id)

        # 坐标越界检查（只对简单形状）
        for attr in ("cx", "cy", "x", "y", "r"):
            val = elem.get(attr)
            if val:
                try:
                    v = float(val)
                    if v < -20 or v > 276:
                        info["coords_out_of_bounds"] += 1
                        break
                except ValueError:
                    pass

        for child in elem:
            stack.append((child, depth + 1))

    return root, info


# ============================================================================
# 颜色工具（CIE76 加权 RGB 近似）
# ============================================================================

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) < 6:
        return (0, 0, 0)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """RGB → CIELAB（简化但比欧氏距离感知准确得多）."""
    # sRGB → linear
    rgb = [x / 255.0 for x in (r, g, b)]
    rgb = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in rgb]
    # linear RGB → XYZ (D65)
    x = rgb[0] * 0.4124564 + rgb[1] * 0.3575761 + rgb[2] * 0.1804375
    y = rgb[0] * 0.2126729 + rgb[1] * 0.7151522 + rgb[2] * 0.0721750
    z = rgb[0] * 0.0193339 + rgb[1] * 0.1191920 + rgb[2] * 0.9503041
    # XYZ → Lab (D65 white point)
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    def f(t):
        delta = 6 / 29
        return t ** (1/3) if t > delta ** 3 else t / (3 * delta ** 2) + 4 / 29
    L = 116 * f(y / yn) - 16
    a = 500 * (f(x / xn) - f(y / yn))
    b_ = 200 * (f(y / yn) - f(z / zn))
    return (L, a, b_)


def _color_distance(c1: str, c2: str) -> float:
    """CIE76 ΔE 感知色差 (0~100+)."""
    try:
        r1, g1, b1 = _hex_to_rgb(c1)
        r2, g2, b2 = _hex_to_rgb(c2)
        L1, a1, b1_ = _rgb_to_lab(r1, g1, b1)
        L2, a2, b2_ = _rgb_to_lab(r2, g2, b2)
        return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1_ - b2_) ** 2)
    except (ValueError, IndexError):
        return 200.0


def _extract_hex_colors(svg: str) -> List[str]:
    return re.findall(r"#[0-9A-Fa-f]{6}\b", svg)


def _extract_prompt_color_names(prompt: str) -> List[str]:
    """从 prompt 提取颜色名，长名优先避重."""
    found = []
    s = prompt.lower()
    for name in COLOR_NAMES:
        if name in s:
            found.append(name)
            s = s.replace(name, " " * len(name))
    return found


def _extract_prompt_hex_colors(prompt: str) -> List[str]:
    return re.findall(r"#[0-9A-Fa-f]{6}\b", prompt)


# ============================================================================
# 形状检测（树遍历 + 正则辅助）
# ============================================================================

def _detect_shapes(svg: str, tree_info: Dict[str, Any]) -> Dict[str, bool]:
    """综合树结构和正则模式检测视觉形状."""
    found: Dict[str, bool] = {}
    tc = tree_info.get("tag_counts", Counter())
    svg_lower = svg.lower()

    # 基于标签类型精确检测
    has_circle = tc.get("circle", 0) > 0
    has_ellipse = tc.get("ellipse", 0) > 0
    has_rect = tc.get("rect", 0) > 0
    has_polygon = tc.get("polygon", 0) > 0
    has_line = tc.get("line", 0) > 0
    has_path = tc.get("path", 0) > 0
    has_polyline = tc.get("polyline", 0) > 0

    # 圆/椭圆类
    found["circle"] = has_circle or has_ellipse
    found["ring"] = has_circle and "fill=\"none\"" in svg_lower

    # 矩形
    found["rectangle"] = has_rect

    # 多边形 → 可能三角形/六边形
    if has_polygon:
        # 通过 points 属性粗略判断
        polygon_points = re.findall(r'<polygon[^>]*points\s*=\s*"([^"]*)"', svg_lower)
        for pts in polygon_points:
            n = len(pts.strip().split())
            if 3 <= n <= 4:
                found["triangle"] = True
            elif 5 <= n <= 8:
                found["hexagon"] = True

    # 线
    found["ray"] = has_line
    found["needle"] = has_line and "stroke" in svg_lower

    # path 精细检测
    if has_path:
        for name, pat in SHAPE_REGEX.items():
            if re.search(pat, svg_lower):
                found[name] = True

    # 综合判断
    found["badge"] = has_circle and has_rect
    found["sun"] = has_circle and ("fill=\"#ff" in svg_lower or "fill=\"#f4a" in svg_lower
                                     or "fill=\"#ffd" in svg_lower or "fill=\"#f2a" in svg_lower
                                     or "fill=\"url(#" in svg_lower)
    found["bee"] = has_circle and has_ellipse
    found["flower"] = found.get("complex_path", False) and has_circle
    found["fish"] = has_ellipse and has_path
    found["column"] = has_line and has_rect
    found["spine"] = has_ellipse and tc.get("ellipse", 0) >= 3
    found["house"] = has_polygon and has_rect
    found["hand"] = found.get("complex_path", False) and len(svg) > 500
    found["leaf"] = found.get("complex_path", False) and ("green" in svg_lower or "5fb" in svg_lower or "4caf" in svg_lower or "7f9a" in svg_lower)
    found["bottle"] = found.get("complex_path", False) and len(svg) > 600
    found["cup"] = has_rect and "rx" in svg_lower
    found["droplet"] = found.get("complex_path", False) and not found.get("swirl", False)
    found["fork"] = has_rect and "rotate" in svg_lower

    return found


# ============================================================================
# 截断 & 异常检测
# ============================================================================

def _detect_anomalies(svg: str, tree_info: Dict[str, Any]) -> List[str]:
    """检测 SVG 中的异常问题，返回问题描述列表."""
    issues = []
    svg_lower = svg.lower()

    # 1. 截断检测：不以 </svg> 结尾
    if not svg_lower.rstrip().endswith("</svg>"):
        issues.append("truncated: missing closing </svg> tag")

    # 2. 空壳检测
    draw_count = tree_info.get("total_drawing_elements", 0)
    if draw_count < 2:
        issues.append(f"near-empty: only {draw_count} drawing element(s)")

    # 3. 失控长度
    if len(svg) > 8000:
        issues.append(f"runaway: excessively long ({len(svg)} chars)")

    # 4. 过短
    if len(svg) < 200:
        issues.append(f"too_short: only {len(svg)} chars")

    # 5. 标签平衡
    for tag in ("g", "defs", "clipPath", "linearGradient", "radialGradient"):
        opens = len(re.findall(f"<{tag}\\b", svg_lower))
        closes = len(re.findall(f"</{tag}>", svg_lower))
        if opens != closes:
            issues.append(f"unbalanced <{tag}>: {opens} open vs {closes} close")

    # 6. 重复 ID
    ids = tree_info.get("ids", [])
    if len(ids) != len(set(ids)):
        duplicates = [i for i, c in Counter(ids).items() if c > 1]
        issues.append(f"duplicate ids: {duplicates[:3]}")

    # 7. 坐标越界
    oob = tree_info.get("coords_out_of_bounds", 0)
    if oob > 0:
        issues.append(f"coordinates out of bounds: {oob} element(s)")

    # 8. 可能包含 markdown/解释文字
    non_tag_text = re.sub(r"<[^>]+>", " ", svg).strip()
    if len(non_tag_text) / max(len(svg), 1) > 0.10:
        issues.append("excessive non-tag text (possible markdown/prose)")

    return issues


# ============================================================================
# 维度一：结构有效性
# ============================================================================

def _score_structural(svg: str) -> Tuple[float, Dict[str, Any]]:
    detail: Dict[str, Any] = {}
    failures: List[str] = []

    if not svg or not svg.strip():
        return 0.0, {"error": "empty SVG"}, ["empty_input"]

    root, tree_info = _parse_svg_tree(svg)
    if root is None:
        # XML 解析失败 — 可能是截断或非法字符
        if "<svg" in svg.lower() and "</svg>" not in svg.lower():
            return 0.03, {"parse_error": "SVG truncated"}, ["xml_parse_failed:truncated"]
        return 0.05, {"parse_error": "XML parse failed"}, ["xml_parse_failed"]

    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag != "svg":
        return 0.08, {"root_tag": tag}, [f"root_is_{tag}_not_svg"]

    score = 1.0

    # xmlns
    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    if not ns or "w3.org" not in ns:
        detail["missing_xmlns"] = True
        score -= 0.20
        failures.append("missing_xmlns")

    # viewBox
    viewbox = root.get("viewBox", "")
    if not viewbox:
        detail["missing_viewBox"] = True
        score -= 0.25
        failures.append("missing_viewBox")
    else:
        parts = viewbox.split()
        if len(parts) >= 4:
            try:
                w, h = float(parts[2]), float(parts[3])
                if w <= 0 or h <= 0:
                    score -= 0.20
                    failures.append("invalid_viewBox")
                elif abs(w - 256) < 5 and abs(h - 256) < 5:
                    detail["viewBox_256x256"] = True  # 正确！
            except ValueError:
                score -= 0.15
                failures.append("viewBox_parse_error")

    # 实质内容
    draw_count = tree_info.get("total_drawing_elements", 0)
    if draw_count < 2:
        score -= 0.30
        failures.append(f"too_few_drawing_elements:{draw_count}")
    elif draw_count < 5:
        score -= 0.10

    # 深度异常
    if tree_info.get("max_depth", 0) > 25:
        score -= 0.10
        failures.append("excessive_nesting")

    detail["drawing_elements"] = draw_count
    detail["element_count"] = tree_info.get("element_count", 0)

    return max(0.0, score), detail, failures


# ============================================================================
# 维度二：设计合规性
# ============================================================================

def _score_compliance(svg: str, tree_info: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    detail: Dict[str, Any] = {}
    failures: List[str] = []
    score = 1.0
    tc = tree_info.get("tag_counts", Counter())

    # 禁止元素
    banned_found = [t for t in BANNED_ELEMENTS if tc.get(t, 0) > 0]
    for b in banned_found:
        score -= 0.35
        failures.append(f"banned_element:{b}")

    # 禁止模式
    for pat in BANNED_PATTERNS:
        if re.search(pat, svg, re.IGNORECASE):
            score -= 0.25
            failures.append(f"banned_pattern:{pat[:30]}")

    # 非白名单标签
    unusual = set(tc.keys()) - ALLOWED_TAGS - {t for t in tc if t.startswith("xml")}
    if unusual:
        detail["unusual_tags"] = list(unusual)
        score -= 0.10 * min(len(unusual), 3)

    # 元素数量
    total_el = sum(tc.values())
    detail["total_elements"] = total_el
    if total_el < 3:
        score -= 0.30
        failures.append("too_few_total_elements")
    elif total_el > 350:
        score -= 0.10

    # 颜色
    hex_colors = _extract_hex_colors(svg)
    unique_colors = len(set(c.lower() for c in hex_colors))
    detail["unique_hex_colors"] = unique_colors
    if unique_colors == 0:
        score -= 0.15
        failures.append("no_colors")
    elif unique_colors > 30:
        score -= 0.10
    elif 3 <= unique_colors <= 12:
        detail["good_palette_size"] = True  # 合理范围

    # 渐变加分
    if tree_info.get("has_gradient"):
        detail["has_gradients"] = True

    return max(0.0, score), detail, failures


# ============================================================================
# 维度三：视觉质量（NEW）
# ============================================================================

def _score_visual(svg: str, tree_info: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    detail: Dict[str, Any] = {}
    failures: List[str] = []
    score = 1.0
    tc = tree_info.get("tag_counts", Counter())

    # 1. 形状多样性
    used_shapes = sum(1 for t in DRAWING_TAGS if tc.get(t, 0) > 0)
    detail["shape_types_used"] = used_shapes
    if used_shapes >= 5:
        detail["rich_shapes"] = True
    elif used_shapes < 2:
        score -= 0.15
        failures.append("poor_shape_diversity")

    # 2. defs + 渐变 → 专业感
    if tree_info.get("has_defs") and tree_info.get("has_gradient"):
        detail["uses_defs_and_gradients"] = True
    elif tree_info.get("has_defs"):
        pass  # 中性
    else:
        score -= 0.05  # 无 defs，可能太简单

    # 3. 坐标越界
    oob = tree_info.get("coords_out_of_bounds", 0)
    detail["coords_out_of_bounds"] = oob
    if oob > 3:
        score -= 0.15
        failures.append(f"many_coords_oob:{oob}")

    # 4. 元素数量与多样性匹配
    total_draw = tree_info.get("total_drawing_elements", 0)
    if 8 <= total_draw <= 80:
        detail["good_drawing_count"] = True
    elif total_draw > 120:
        score -= 0.05

    return max(0.0, score), detail, failures


# ============================================================================
# 维度四：prompt-SVG 对齐度（核心）
# ============================================================================

def _score_alignment(prompt: str, svg: str, tree_info: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    detail: Dict[str, Any] = {}
    failures: List[str] = []
    sub: Dict[str, float] = {}
    prompt_lower = prompt.lower()
    svg_lower = svg.lower()

    # --- 4.1 颜色对齐（阶梯评分） ---
    prompt_hex = _extract_prompt_hex_colors(prompt)
    svg_hex = [c.lower() for c in _extract_hex_colors(svg)]

    if prompt_hex:
        matched = len(set(c.lower() for c in prompt_hex) & set(svg_hex))
        color_match = matched / len(prompt_hex)
        detail["color_mode"] = "hex"
    else:
        named_colors = _extract_prompt_color_names(prompt)
        detail["prompt_color_names"] = named_colors
        if named_colors:
            hits = 0
            for name in named_colors:
                ref = COLOR_NAME_TO_HEX.get(name)
                if ref and ref != "none":
                    for sh in svg_hex:
                        if _color_distance(ref, sh) < 35:  # CIE76 < 35 = 感知接近
                            hits += 1
                            break
            color_match = hits / len(named_colors)
        else:
            color_match = 0.5  # 无颜色信息
        detail["color_mode"] = "name"

    # 阶梯化
    if color_match >= 0.80:
        color_score = 0.90 + 0.10 * (color_match - 0.80) / 0.20
    elif color_match >= 0.50:
        color_score = 0.60 + 0.30 * (color_match - 0.50) / 0.30
    elif color_match >= 0.20:
        color_score = 0.30 + 0.30 * (color_match - 0.20) / 0.30
    else:
        color_score = 0.30 * color_match / 0.20
    sub["color_match_raw"] = round(color_match, 3)
    sub["color_score"] = round(color_score, 3)

    # --- 4.2 形状对齐 ---
    shapes_found = _detect_shapes(svg, tree_info)
    prompt_shapes: Set[str] = set()
    for shape_name in SHAPE_REGEX:
        if shape_name in prompt_lower:
            prompt_shapes.add(shape_name)
    # 也加入基于语义组的形状
    for group_name, group_info in SEMANTIC_GROUPS.items():
        if any(kw in prompt_lower for kw in group_info["keywords"]):
            for indicator in group_info["svg_indicators"]:
                prompt_shapes.add(indicator)

    detail["prompt_shapes_inferred"] = sorted(prompt_shapes)
    detail["svg_shapes_detected"] = sorted(shapes_found.keys())

    if prompt_shapes:
        hits = sum(1 for s in prompt_shapes if shapes_found.get(s, False))
        shape_match = hits / len(prompt_shapes)
    else:
        shape_match = 0.5
    # 阶梯化
    if shape_match >= 0.70:
        shape_score = 0.85 + 0.15 * (shape_match - 0.70) / 0.30
    elif shape_match >= 0.40:
        shape_score = 0.55 + 0.30 * (shape_match - 0.40) / 0.30
    elif shape_match >= 0.15:
        shape_score = 0.25 + 0.30 * (shape_match - 0.15) / 0.25
    else:
        shape_score = 0.25 * shape_match / 0.15
    sub["shape_match_raw"] = round(shape_match, 3)
    sub["shape_score"] = round(shape_score, 3)

    # --- 4.3 复杂度对齐 ---
    prompt_words = len(prompt.split())
    svg_len = len(svg)
    if svg_len < 150:
        complexity_score = 0.15   # 几乎肯定是不完整/截断
        failures.append("svg_too_short_for_alignment")
    elif svg_len > 8000:
        complexity_score = 0.30   # 失控
        failures.append("svg_too_long_for_alignment")
    else:
        ratio = svg_len / max(prompt_words * 25, 1)
        if 0.3 <= ratio <= 2.5:
            complexity_score = 0.95
        elif 0.15 <= ratio <= 4.0:
            complexity_score = 0.70
        else:
            complexity_score = 0.40
    sub["complexity"] = round(complexity_score, 3)
    detail["svg_len"] = svg_len
    detail["prompt_words"] = prompt_words

    # --- 4.4 非 SVG 文本 ---
    text_only = re.sub(r"<[^>]+>", " ", svg).strip()
    non_tag_ratio = len(text_only) / max(len(svg), 1)
    detail["non_tag_text_ratio"] = round(non_tag_ratio, 4)
    if non_tag_ratio > 0.10:
        text_penalty = min(0.40, (non_tag_ratio - 0.05) * 4)
    else:
        text_penalty = 0.0
    sub["text_clean"] = round(1.0 - text_penalty, 3)

    # --- 4.5 语义组覆盖 ---
    semantic_hit = 0
    semantic_total = 0
    for group_name, group_info in SEMANTIC_GROUPS.items():
        if any(kw in prompt_lower for kw in group_info["keywords"]):
            semantic_total += 1
            if any(shapes_found.get(ind, False) for ind in group_info["svg_indicators"]):
                semantic_hit += 1
    detail["semantic_groups_matched"] = f"{semantic_hit}/{semantic_total}"
    if semantic_total > 0:
        semantic_score = semantic_hit / semantic_total
    else:
        semantic_score = 0.5
    sub["semantic"] = round(semantic_score, 3)

    # --- 4.6 设计品质加分（缩小幅度，避免顶部聚集） ---
    quality_bonus = 0.0
    tc = tree_info.get("tag_counts", Counter())
    if tree_info.get("has_defs") and tree_info.get("has_gradient"):
        quality_bonus += 0.04
    shape_types = sum(1 for t in DRAWING_TAGS if tc.get(t, 0) > 0)
    if shape_types >= 4:
        quality_bonus += 0.03
    if 6 <= tree_info.get("total_drawing_elements", 0) <= 60:
        quality_bonus += 0.03
    sub["quality_bonus"] = round(quality_bonus, 3)

    # --- 综合 ---
    raw = (
        0.30 * color_score
        + 0.25 * shape_score
        + 0.10 * complexity_score
        + 0.10 * (1.0 - text_penalty)
        + 0.15 * semantic_score
        + 0.05  # 极小基础分（v1 是 0.15）
        + quality_bonus
    )
    detail["sub_scores"] = {k: round(v, 4) for k, v in sub.items()}

    return min(1.0, max(0.0, raw)), detail, failures


# ============================================================================
# 主接口
# ============================================================================

def score(prompt: str, svg_output: str) -> Tuple[float, Dict[str, Any]]:
    """评估 SVG Logo 质量，返回 (总分, 详细信息)."""
    if not svg_output or not svg_output.strip():
        return 0.0, {"error": "empty output", "failure_reasons": ["empty_output"]}

    svg = svg_output.strip()
    svg_match = re.search(r"<svg[\s\S]*?</svg>", svg, re.IGNORECASE)
    if svg_match:
        svg = svg_match.group(0)
    elif "<svg" in svg:
        svg = svg[svg.index("<svg"):] + "</svg>"
    else:
        return 0.03, {"error": "no <svg> tag found", "failure_reasons": ["no_svg_tag"]}

    # 解析一次，复用
    _, tree_info = _parse_svg_tree(svg)

    # 异常检测
    anomalies = _detect_anomalies(svg, tree_info)

    # 四维打分
    s1, d1, f1 = _score_structural(svg)
    s2, d2, f2 = _score_compliance(svg, tree_info)
    s3, d3, f3 = _score_visual(svg, tree_info)
    s4, d4, f4 = _score_alignment(prompt, svg, tree_info)

    all_failures = anomalies + f1 + f2 + f3 + f4

    total = (
        WEIGHTS["structural"] * s1
        + WEIGHTS["compliance"] * s2
        + WEIGHTS["visual"] * s3
        + WEIGHTS["alignment"] * s4
    )

    detail = {
        "total": round(total, 4),
        "structural": {"score": round(s1, 4), **d1},
        "compliance": {"score": round(s2, 4), **d2},
        "visual":     {"score": round(s3, 4), **d3},
        "alignment":  {"score": round(s4, 4), **d4},
        "failure_reasons": all_failures[:12],  # 最多 12 条
    }

    return total, detail


# ============================================================================
# 批量评估
# ============================================================================

def score_batch(samples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results = []
    for i, sample in enumerate(samples):
        prompt = sample.get("prompt", "")
        svg_output = sample.get("svg_output", "")
        total, detail = score(prompt, svg_output)
        results.append({
            "index": i, "prompt": prompt, "svg_output": svg_output,
            "score": total, "detail": detail,
        })
    return results


def score_from_jsonl(
    jsonl_path: str, svg_field: str = "svg_output",
    prompt_field: str = "prompt",
) -> List[Dict[str, Any]]:
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append({
                "prompt": record.get(prompt_field, ""),
                "svg_output": record.get(svg_field, ""),
            })
    return score_batch(samples)


# ============================================================================
# Sonnet-as-judge（增强版：缓存 + 重试）
# ============================================================================

_JUDGE_CACHE: Dict[str, Tuple[float, Dict]] = {}


def _make_cache_key(prompt: str, svg: str) -> str:
    return hashlib.md5((prompt[:500] + svg[:2000]).encode()).hexdigest()


def score_with_sonnet(
    prompt: str,
    svg_output: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 2,
) -> Tuple[float, Dict[str, Any]]:
    """使用 Claude Sonnet 评判 SVG 质量（带缓存和重试）."""
    import os

    # 缓存命中的话直接返回
    cache_key = _make_cache_key(prompt, svg_output)
    if cache_key in _JUDGE_CACHE:
        return _JUDGE_CACHE[cache_key]

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        # 回退到规则评分
        return score(prompt, svg_output)

    from anthropic import Anthropic
    client = Anthropic(api_key=key)

    judge_prompt = f"""You are a strict SVG logo quality judge. Score this SVG against its original description.

Rate EACH dimension from 0 (worst) to 10 (best), using the FULL 0-10 range:
1. **structural** (0-10): Is it well-formed XML? Has xmlns + correct viewBox="0 0 256 256"? No parse errors?
2. **compliance** (0-10): Only vector primitives? No <image>/<script>/external refs? Proper palette size?
3. **visual** (0-10): Are elements within 0-256 bounds? Good shape diversity? Composition balanced?
4. **alignment** (0-10): Does the SVG ACCURATELY draw what the prompt describes? Colors, shapes, layout MUST match.

IMPORTANT: Reserve 9-10 for near-perfect outputs. Use 5-7 for decent but flawed. Use 1-4 for poor.

Original prompt:
{prompt[:1000]}

SVG to judge:
{svg_output[:3000]}

Return ONLY a JSON object:
{{"structural": <int 0-10>, "compliance": <int 0-10>, "visual": <int 0-10>, "alignment": <int 0-10>, "comment": "<1 sentence>"}}"""

    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            text = resp.content[0].text.strip()
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                result = json.loads(json_match.group(0))
                total = (
                    result.get("structural", 5) * WEIGHTS["structural"]
                    + result.get("compliance", 5) * WEIGHTS["compliance"]
                    + result.get("visual", 5) * WEIGHTS["visual"]
                    + result.get("alignment", 5) * WEIGHTS["alignment"]
                ) / 10.0
                output = (total, {"sonnet_judge": result, "cache_key": cache_key})
                _JUDGE_CACHE[cache_key] = output
                return output
        except Exception:
            if attempt == max_retries:
                break

    # 重试耗尽，回退
    total, detail = score(prompt, svg_output)
    _JUDGE_CACHE[cache_key] = (total, detail)
    return total, detail


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python reward.py <results.jsonl>")
        print("       python reward.py --demo")
        sys.exit(1)

    if sys.argv[1] == "--demo":
        demo_prompt = "A simple red circle on white background"
        demo_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect width="256" height="256" fill="#ffffff"/>
  <circle cx="128" cy="128" r="80" fill="#ff0000"/>
</svg>'''
        total, detail = score(demo_prompt, demo_svg)
        print(json.dumps(detail, indent=2, ensure_ascii=False))
        print(f"\nTotal reward: {total:.4f}")
    else:
        path = sys.argv[1]
        results = score_from_jsonl(path)
        output_path = path.replace(".jsonl", "_reward.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        avg = sum(r["score"] for r in results) / len(results) if results else 0
        print(f"Scored {len(results)} samples → {output_path}")
        print(f"Average reward: {avg:.4f}")
