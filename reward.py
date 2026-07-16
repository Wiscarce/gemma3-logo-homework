"""
SVG Logo Reward Function — 课程作业 Part B

评估维度:
  1. SVG 结构有效性 (structural validity) — XML 是否合法、关键属性是否存在
  2. 设计规则合规性 (design-rule compliance) — 是否遵守 SVG 生成规则
  3. 提示词对齐度 (prompt–SVG alignment) — 生成内容是否符合 prompt 描述

返回:
  float 总分 (0.0 ~ 1.0)
  dict  各维度分项得分

用法:
  from reward import score
  total, detail = score(prompt="...", svg_output="<svg ...>")
"""

from __future__ import annotations

import re
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 可配置权重
# ---------------------------------------------------------------------------
WEIGHTS = {
    "structural": 0.30,
    "compliance": 0.25,
    "alignment":  0.45,   # 对齐度最重要
}

# ---------------------------------------------------------------------------
# 禁止使用的 SVG 元素 / 属性
# ---------------------------------------------------------------------------
BANNED_ELEMENTS = {"image", "script", "foreignObject", "use", "iframe", "style"}
BANNED_PATTERNS = [
    r"href\s*=\s*[\"\']",
    r"xlink:href\s*=",
    r"javascript\s*:",
    r"<script",
    r"on\w+\s*=",
]

ALLOWED_TAGS = {
    "svg", "defs", "g", "path", "circle", "ellipse",
    "rect", "line", "polygon", "polyline",
    "linearGradient", "radialGradient", "stop",
    "clipPath", "filter", "feGaussianBlur", "feMerge", "feMergeNode",
    "mask", "pattern",
    "title", "desc", "metadata",
}

# ---------------------------------------------------------------------------
# 颜色名称 → 典型 hex 值映射 (用于对齐度计算)
# ---------------------------------------------------------------------------
COLOR_NAME_TO_HEX = {
    "navy": "#1b3a5c", "blue": "#2a6fdb", "teal": "#2ec4b6", "deep teal": "#0f5c56",
    "green": "#4caf50", "forest green": "#1f4d33", "forest": "#1f4d33",
    "sage green": "#cddfc4", "sage": "#cddfc4",
    "orange": "#ff6b35", "warm orange": "#f2994a", "coral": "#ff6f5e",
    "gold": "#f2a93b", "golden": "#d4a017", "mustard": "#d2a531",
    "yellow": "#ffd93b", "mango yellow": "#ffc145",
    "red": "#d64541", "burgundy": "#8f2438",
    "cream": "#fbf3e3", "white": "#ffffff", "off-white": "#fff8f0",
    "charcoal": "#2d2a26", "black": "#1b1b1e",
    "amber": "#b85a18", "deep amber": "#b85a18",
    "slate blue": "#2a4759", "slate": "#2a4759",
    "sky blue": "#8ecae6", "violet": "#a239ff", "cyan": "#22e5ff",
    "brown": "#5c4326", "gray": "#888888", "grey": "#888888",
    "pink": "#ff9eb5", "purple": "#7c3aed", "lavender": "#c4b5fd",
    "indigo": "#6366f1", "olive": "#808000", "mint": "#98fb98",
    "peach": "#ffdab9", "maroon": "#800000",
}
COLOR_NAMES = list(COLOR_NAME_TO_HEX.keys())

# ---------------------------------------------------------------------------
# 视觉形状概念 → 可检测的 SVG 元素模式
# ---------------------------------------------------------------------------
SHAPE_PATTERNS = {
    "circle": [r"<circle\b"],
    "hexagon": [r'<polygon\b[^>]*\bpoints\s*=\s*"[^"]{40,}'],
    "star": [r'<polygon\b', r'<path\b[^>]*\bd\s*=\s*"[^"]{30,}'],
    "triangle": [r'<polygon\b[^>]*\bpoints\s*=\s*"[^"]{1,20}"'],
    "rectangle": [r"<rect\b"],
    "shield": [r'<path\b[^>]*\bd\s*=\s*"[^"]{30,}'],
    "badge": [r"<circle\b", r"<rect\b.*\brx\s*="],
    "sun": [r'<circle\b[^>]*\bfill\s*=\s*"[^"]*(?:ff[cd]|orange|gold|yellow|f4a|ffd)'],
    "wave": [r'<path\b[^>]*\bd\s*=\s*"[^"]*[cCqQ]'],
    "leaf": [r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:C|Q).*(?:C|Q)'],
    "spiral": [r'<path\b[^>]*\bd\s*=\s*"[^"]*[cC].*[cC].*[cC]'],
    "swirl": [r'<path\b[^>]*\bd\s*=\s*"[^"]*[cC].*[cC]'],
    "droplet": [r'<path\b[^>]*\bd\s*=\s*"[^Mi][^"]{20,}'],
    "bubble": [r"<circle\b[^>]*\br\s*=\s*\"[0-5]"],
    "ray": [r"<line\b", r"<rect\b.*rotate"],
    "ribbon": [r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:[cC].*){3,}'],
    "arch": [r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:A|a)\b'],
    "hand": [r'<path\b[^>]*\bd\s*=\s*"[^M][^"]{50,}'],
    "arrow": [r'<polygon\b[^>]*\bpoints\s*=\s*"[^"]{1,25}"'],
    "tree": [r'<polygon\b', r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:L|M)[^"]*(?:L|M)'],
    "flower": [r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:C|Q).*(?:C|Q).*(?:C|Q)'],
    "bottle": [r'<path\b[^>]*\bd\s*=\s*"[^M][^"]{30,}'],
    "cup": [r'<rect\b.*\brx\s*=', r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:Q|C)'],
    "mug": [r'<rect\b', r'<path\b'],
    "house": [r'<polygon\b', r'<rect\b'],
    "roof": [r'<polygon\b[^>]*\bpoints\s*=\s*"[^"]{1,25}"'],
    "sparkle": [r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:L[^"]*){4,}'],
    "fish": [r'<ellipse\b', r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:C|Q)'],
    "bee": [r'<ellipse\b', r'<circle\b'],
    "column": [r"<line\b", r"<rect\b"],
    "spine": [r"<ellipse\b"],
    "scissors": [r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:M[^"]*L[^"]*){2,}'],
    "needle": [r"<line\b[^>]*stroke"],
    "fork": [r"<rect\b.*rotate"],
    "brush": [r'<rect\b.*rotate', r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:c|C)'],
    "paint": [r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:c|C).*(?:c|C)'],
    "note": [r'<ellipse\b', r'<path\b'],
    "swoosh": [r'<path\b[^>]*\bd\s*=\s*"[^"]*(?:c|C|q|Q)'],
    "ring": [r'<circle\b[^>]*fill\s*=\s*"none"'],
}


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _color_distance(c1: str, c2: str) -> float:
    """两个 hex 颜色之间的欧几里德距离 (0~441)."""
    try:
        r1, g1, b1 = _hex_to_rgb(c1)
        r2, g2, b2 = _hex_to_rgb(c2)
        return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
    except (ValueError, IndexError):
        return 442.0


def _extract_hex_colors(svg: str) -> List[str]:
    return re.findall(r"#[0-9A-Fa-f]{6}\b", svg)


def _extract_prompt_color_names(prompt: str) -> List[str]:
    """从 prompt 中提取颜色名称."""
    found = []
    prompt_lower = prompt.lower()
    # 按长度排序, 先匹配长的 (如 "deep forest green" > "green")
    sorted_names = sorted(COLOR_NAMES, key=len, reverse=True)
    for name in sorted_names:
        if name in prompt_lower:
            found.append(name)
            prompt_lower = prompt_lower.replace(name, "")  # 避免重复匹配
    return found


def _extract_prompt_hex_colors(prompt: str) -> List[str]:
    return re.findall(r"#[0-9A-Fa-f]{6}\b", prompt)


def _extract_svg_tag_structure(svg: str) -> Dict[str, bool]:
    """检测 SVG 中存在的视觉结构."""
    found = {}
    svg_lower = svg.lower()
    for shape, patterns in SHAPE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, svg_lower):
                found[shape] = True
                break
    return found


def _extract_tag_count(svg: str) -> Counter:
    tags = re.findall(r"<(\w+)", svg)
    return Counter(tags)


def _svg_to_etree(svg: str) -> Tuple[Optional[ET.Element], Optional[str]]:
    try:
        cleaned = svg.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        root = ET.fromstring(cleaned)
        return root, None
    except ET.ParseError as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# 维度一: 结构有效性
# ---------------------------------------------------------------------------

def _score_structural(svg: str) -> Tuple[float, Dict[str, Any]]:
    detail: Dict[str, Any] = {}

    if not svg or not svg.strip():
        return 0.0, {"error": "empty SVG"}

    root, err = _svg_to_etree(svg)
    if root is None:
        detail["parse_error"] = err
        return 0.05, detail

    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag != "svg":
        detail["root_tag"] = tag
        return 0.1, detail

    score = 1.0

    # xmlns 检查
    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    if not ns or "w3.org" not in ns:
        detail["missing_xmlns"] = True
        score -= 0.25

    # viewBox 检查
    viewbox = root.get("viewBox", "")
    if not viewbox:
        detail["missing_viewBox"] = True
        score -= 0.25
    else:
        parts = viewbox.split()
        if len(parts) >= 4:
            try:
                w, h = float(parts[2]), float(parts[3])
                if w <= 0 or h <= 0:
                    detail["invalid_viewBox"] = viewbox
                    score -= 0.2
            except ValueError:
                score -= 0.15

    # 检查 SVG 是否有实质内容 (不只是空壳)
    tag_count = _extract_tag_count(svg)
    content_tags = sum(
        v for k, v in tag_count.items()
        if k not in {"svg", "defs", "title", "desc", "metadata"}
    )
    if content_tags < 2:
        detail["empty_shell"] = True
        score -= 0.3

    return max(0.0, score), detail


# ---------------------------------------------------------------------------
# 维度二: 设计规则合规性
# ---------------------------------------------------------------------------

def _score_compliance(svg: str) -> Tuple[float, Dict[str, Any]]:
    detail: Dict[str, Any] = {}
    score = 1.0

    tag_count = _extract_tag_count(svg)

    # 禁止元素
    banned_found = [t for t in BANNED_ELEMENTS if tag_count.get(t, 0) > 0]
    if banned_found:
        detail["banned_elements"] = banned_found
        score -= 0.35 * len(banned_found)

    # 禁止模式
    banned_pats = [p for p in BANNED_PATTERNS if re.search(p, svg, re.IGNORECASE)]
    if banned_pats:
        detail["banned_patterns"] = banned_pats
        score -= 0.25 * len(banned_pats)

    # 标签检查
    unusual = set(tag_count.keys()) - ALLOWED_TAGS - {t for t in tag_count if t.startswith("xml")}
    if unusual:
        detail["unusual_tags"] = list(unusual)
        score -= 0.1 * min(len(unusual), 3)

    # 元素总数
    total_el = sum(tag_count.values())
    detail["total_elements"] = total_el
    if total_el < 5:
        score -= 0.25
    elif total_el > 300:
        score -= 0.1

    # 颜色使用
    colors = _extract_hex_colors(svg)
    unique_colors = len(set(c.lower() for c in colors))
    detail["unique_hex_colors"] = unique_colors
    if unique_colors == 0:
        score -= 0.15
    elif unique_colors > 25:
        score -= 0.1

    # 使用 gradien 是好迹象
    if any(t in tag_count for t in ["linearGradient", "radialGradient"]):
        detail["has_gradients"] = True

    return max(0.0, score), detail


# ---------------------------------------------------------------------------
# 维度三: prompt–SVG 对齐度
# ---------------------------------------------------------------------------

def _score_alignment(prompt: str, svg: str) -> Tuple[float, Dict[str, Any]]:
    detail: Dict[str, Any] = {}
    sub_scores: Dict[str, float] = {}
    svg_lower = svg.lower()
    prompt_lower = prompt.lower()

    # --- 3.1 颜色对齐 ---
    prompt_hex = _extract_prompt_hex_colors(prompt)
    svg_hex = [c.lower() for c in _extract_hex_colors(svg)]

    if prompt_hex:
        # prompt 里有明确的 hex 颜色 → 检查 SVG 是否使用了它们
        prompt_hex_set = set(c.lower() for c in prompt_hex)
        svg_hex_set = set(svg_hex)
        color_match = len(prompt_hex_set & svg_hex_set) / len(prompt_hex_set)
        detail["prompt_hex_colors"] = list(prompt_hex_set)
        detail["svg_hex_colors_found"] = list(svg_hex_set & prompt_hex_set)
    else:
        # prompt 使用颜色名称 → 检查 SVG 中是否有匹配的 hex
        named_colors = _extract_prompt_color_names(prompt)
        detail["prompt_color_names"] = named_colors
        if named_colors:
            matches = 0
            for name in named_colors:
                ref_hex = COLOR_NAME_TO_HEX.get(name)
                if ref_hex:
                    # 检查 SVG 中是否有足够接近的颜色
                    for sh in svg_hex:
                        if _color_distance(ref_hex, sh) < 60:
                            matches += 1
                            break
            color_match = matches / len(named_colors)
        else:
            color_match = 0.6  # 无颜色信息时给中性偏高分
    sub_scores["color_match"] = color_match

    # --- 3.2 形状/视觉元素对齐 ---
    svg_structure = _extract_svg_tag_structure(svg)
    detail["svg_structures_detected"] = list(svg_structure.keys())

    # 从 prompt 推断应该出现的视觉元素
    prompt_shapes_infer: Dict[str, bool] = {}
    for shape in SHAPE_PATTERNS:
        # 检查 prompt 中是否提到该形状
        if shape in prompt_lower:
            prompt_shapes_infer[shape] = True
    detail["prompt_shapes_inferred"] = list(prompt_shapes_infer.keys())

    if prompt_shapes_infer:
        shape_hits = sum(
            1 for s in prompt_shapes_infer if svg_structure.get(s, False)
        )
        shape_match = shape_hits / len(prompt_shapes_infer)
    else:
        shape_match = 0.5
    sub_scores["shape_match"] = shape_match

    # --- 3.3 复杂度对齐 ---
    prompt_words = len(prompt.split())
    svg_len = len(svg)
    # prompt 越长 (越复杂的描述) → 期望更长的 SVG
    expected_svg_len = prompt_words * 30  # 粗略估算
    complexity_ratio = min(svg_len / max(expected_svg_len, 1), 3.0)
    if 0.3 < complexity_ratio < 2.5:
        complexity_score = 1.0
    elif complexity_ratio < 0.1:
        complexity_score = 0.2  # SVG 太短, 可能什么都没画
    else:
        complexity_score = 0.7
    sub_scores["complexity"] = complexity_score
    detail["svg_len"] = svg_len
    detail["prompt_word_count"] = prompt_words

    # --- 3.4 非 SVG 文本惩罚 ---
    text_only = re.sub(r"<[^>]+>", " ", svg).strip()
    non_tag_ratio = len(text_only) / max(len(svg), 1)
    detail["non_tag_text_ratio"] = round(non_tag_ratio, 4)
    if non_tag_ratio > 0.08:
        text_penalty = min(0.5, (non_tag_ratio - 0.05) * 5)
    else:
        text_penalty = 0.0
    sub_scores["text_clean"] = 1.0 - text_penalty

    # --- 3.5 设计品质加分 ---
    quality_bonus = 0.0
    tag_count = _extract_tag_count(svg)
    # 有 defs + gradient → 更精致
    if "<defs>" in svg and ("linearGradient" in tag_count or "radialGradient" in tag_count):
        quality_bonus += 0.1
    # 有混合形状 (path + circle + rect 等) → 更丰富
    shape_tag_types = {"path", "circle", "ellipse", "rect", "polygon", "line", "polyline"}
    used_shape_types = sum(1 for t in shape_tag_types if tag_count.get(t, 0) > 0)
    if used_shape_types >= 3:
        quality_bonus += 0.05
    if used_shape_types >= 5:
        quality_bonus += 0.05
    sub_scores["quality_bonus"] = quality_bonus

    # --- 综合 ---
    raw = (
        0.30 * color_match
        + 0.25 * shape_match
        + 0.15 * complexity_score
        + 0.15 * (1.0 - text_penalty)
        + 0.15  # 基础分
        + quality_bonus
    )
    detail["sub_scores"] = {k: round(v, 4) for k, v in sub_scores.items()}

    return min(1.0, max(0.0, raw)), detail


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def score(prompt: str, svg_output: str) -> Tuple[float, Dict[str, Any]]:
    if not svg_output or not svg_output.strip():
        return 0.0, {"error": "empty output"}

    svg = svg_output.strip()
    svg_match = re.search(r"<svg[\s\S]*?</svg>", svg, re.IGNORECASE)
    if svg_match:
        svg = svg_match.group(0)
    elif "<svg" in svg:
        svg = svg[svg.index("<svg"):] + "</svg>"
    else:
        return 0.05, {"error": "no <svg> tag found"}

    s1, d1 = _score_structural(svg)
    s2, d2 = _score_compliance(svg)
    s3, d3 = _score_alignment(prompt, svg)

    total = (
        WEIGHTS["structural"] * s1
        + WEIGHTS["compliance"] * s2
        + WEIGHTS["alignment"] * s3
    )

    detail = {
        "total": round(total, 4),
        "structural": {"score": round(s1, 4), **d1},
        "compliance": {"score": round(s2, 4), **d2},
        "alignment":  {"score": round(s3, 4), **d3},
    }

    return total, detail


# ---------------------------------------------------------------------------
# 批量评估
# ---------------------------------------------------------------------------

def score_batch(samples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results = []
    for i, sample in enumerate(samples):
        prompt = sample.get("prompt", "")
        svg_output = sample.get("svg_output", "")
        total, detail = score(prompt, svg_output)
        results.append({
            "index": i,
            "prompt": prompt,
            "svg_output": svg_output,
            "score": total,
            "detail": detail,
        })
    return results


def score_from_jsonl(
    jsonl_path: str,
    svg_field: str = "svg_output",
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


# ---------------------------------------------------------------------------
# Sonnet-as-judge (可选, 更高质量的 prompt-alignment 评分)
# ---------------------------------------------------------------------------

def score_with_sonnet(
    prompt: str,
    svg_output: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> Tuple[float, Dict[str, Any]]:
    """
    使用 Anthropic Claude Sonnet 作为评判者进行打分。
    需要: pip install anthropic
    需要: ANTHROPIC_API_KEY 环境变量或传入 api_key
    """
    import os
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("需要 ANTHROPIC_API_KEY")

    from anthropic import Anthropic
    client = Anthropic(api_key=key)

    judge_prompt = f"""Evaluate this SVG logo against its original description.

Rate each dimension 0 (worst) to 10 (best):
1. structural: Well-formed XML? Correct viewBox / xmlns?
2. compliance: Vector primitives only? Good palette? No banned elements?
3. alignment: Does it accurately draw what the prompt describes?

Prompt: {prompt}

SVG: {svg_output}

Return ONLY JSON: {{"structural": <int>, "compliance": <int>, "alignment": <int>, "comment": "<brief>"}}"""

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        text = resp.content[0].text.strip()
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            result = json.loads(json_match.group(0))
            total = (
                result.get("structural", 0) * WEIGHTS["structural"]
                + result.get("compliance", 0) * WEIGHTS["compliance"]
                + result.get("alignment", 0) * WEIGHTS["alignment"]
            ) / 10.0
            return total, {"sonnet_judge": result}
    except Exception:
        pass
    return score(prompt, svg_output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
