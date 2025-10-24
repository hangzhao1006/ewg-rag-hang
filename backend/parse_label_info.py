# -*- coding: utf-8 -*-
"""
把 EWG 爬下来的 label_information 拆分为：
- ingredients_from_packaging（列表+整段）
- directions_from_packaging（列表+整段）
- warnings_from_packaging（列表+整段）

输入： --in_jsonl  （来自 crawl_ewg_face_safe.py 的输出）
输出： --out_jsonl 结构化 JSONL
      --out_csv    扁平视图（便于快速看）

安装：pip install beautifulsoup4 lxml pandas regex
"""

import json
import argparse
import pandas as pd
from bs4 import BeautifulSoup, Tag
import regex as re
from typing import Dict, List, Tuple, Optional
import requests
import time
from urllib.parse import urljoin

# ---- 词表（可按需扩展） ----
HEADINGS = {
    "ingredients": [
        r"ingredients?\s*from\s*packaging",
        r"ingredients?\b",
        r"(?:成分|配方|配料)(?:（包装）|（来自包装）)?",
    ],
    "directions": [
        r"directions?\s*from\s*packaging",
        r"directions?\b",
        r"(?:使用方法|用法|用法用量|使用说明)(?:（来自包装）)?",
        r"(?:How to use)",
    ],
    "warnings": [
        r"warnings?\s*from\s*packaging",
        r"warnings?\b",
        r"(?:警告|注意事项|注意|禁忌)(?:（来自包装）)?",
        r"(?:Caution|Precautions?)",
    ],
}
# 编译为大小写不敏感
HEADINGS = {k: [re.compile(pat, re.I) for pat in v] for k, v in HEADINGS.items()}


def norm_text(s: str) -> str:
    return re.sub(r"[ \t\r\f]+", " ", (s or "").strip())


def split_by_html_headings(html: str) -> Dict[str, Dict[str, str]]:
    """
    优先用 HTML 标题切分：找到 h2/h3/h4/p 中看起来像小标题的节点，
    把其后到下一个标题之间的内容归为该段。
    返回 {section: {"text": "...", "html": "..."}}
    """
    out = {"ingredients": {}, "directions": {}, "warnings": {}}
    if not html:
        return out

    soup = BeautifulSoup(html, "lxml")

    # 可能小标题出现在 h2/h3/h4 或 strong 的 p/li 中
    candidate_nodes: List[Tag] = []
    for tag in soup.find_all(["h2", "h3", "h4", "p", "li"]):
        txt = norm_text(tag.get_text(" ", strip=True))
        if not txt:
            continue
        # 只把“像标题”的拿出来：强制前缀匹配更稳
        for sec, pats in HEADINGS.items():
            if any(p.match(txt) for p in pats):
                candidate_nodes.append(tag)
                break

    if not candidate_nodes:
        return out

    # 依序收集每个标题后的兄弟节点直到下一个标题
    for idx, node in enumerate(candidate_nodes):
        title_text = norm_text(node.get_text(" ", strip=True))
        section: Optional[str] = None
        for sec, pats in HEADINGS.items():
            if any(p.match(title_text) for p in pats):
                section = sec
                break
        if not section:
            continue

        # 收集区间
        parts_html, parts_txt = [], []
        # 先包含“标题节点后面的内容”，不把标题自身重复加入
        cur = node.next_sibling
        while cur and cur not in candidate_nodes:
            if isinstance(cur, Tag):
                # 如果遇到下一个“标题型节点”，break（容错：再次判定）
                if cur in candidate_nodes:
                    break
                parts_html.append(str(cur))
                parts_txt.append(cur.get_text(" ", strip=True))
            cur = cur.next_sibling

        # 合成（只保留纯文本）
        out[section] = {
            # 保留段落的纯文本用于最终输出
            "text": norm_text(" ".join([t for t in parts_txt if t])),
            # 同时保留该段对应的原始 HTML 片段，供后续优先从 <li> 提取 bullets
            "html": "".join(parts_html),
        }

    return out


def bullets_from_html(html: str) -> List[str]:
    """
    仅从 HTML 中提取真实的 <li> 项目作为 bullets。
    不再退化到按 <p>/<br> 拆分——这样可以避免把原本为一段的纯文本错误地当成列表项。
    返回标准化后的文本列表（可能为空）。
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    items: List[str] = []
    for li in soup.select("li"):
        t = norm_text(li.get_text(" ", strip=True))
        if t:
            items.append(t)
    return items


def parse_ingredient_concerns_list(html: str) -> List[Dict[str, str]]:
    """
    从 Ingredient Concerns 的 HTML 中提取关注点列表。
    返回列表，每项为 {"level": "HIGH|MODERATE|LOW|...", "concern": "Cancer"}
    """
    out: List[Dict[str, str]] = []
    if not html:
        return out
    soup = BeautifulSoup(html, "lxml")

    # 尝试在页面中查找 li.concern
    for li in soup.select("li.concern"):
        # 取 concern 文本
        concern_el = li.select_one(".concern-text")
        concern = norm_text(concern_el.get_text(" ", strip=True)) if concern_el else None

        # 取 level 文本或由类名判定
        level_el = li.select_one(".level")
        level = None
        if level_el:
            txt = norm_text(level_el.get_text(" ", strip=True))
            m = re.search(r"(high|moderate|low)", txt, re.I)
            if m:
                level = m.group(1).upper()
        # 若上面未识别，尝试检查内部用于表示级别的类名
        if not level:
            # 查找带 concern-high/concern-moderate/concern-low 的元素
            cls_el = li.select_one(".concern-high, .concern-moderate, .concern-low")
            if cls_el and cls_el.has_attr("class"):
                cls = " ".join(cls_el.get("class", []))
                if "concern-high" in cls:
                    level = "HIGH"
                elif "concern-moderate" in cls:
                    level = "MODERATE"
                elif "concern-low" in cls:
                    level = "LOW"

        if concern or level:
            out.append({"level": level or "", "concern": concern or ""})
    # 如果上面没有命中，再尝试表格/通用结构：查找包含 "CONCERNS" 文本的单元格或标签
    if out:
        return out

    # 查找显式标签（例如 <td>CONCERNS</td>）并取其相邻单元格的内容
    candidates = []
    for tag in soup.find_all(text=re.compile(r"\bCONCERNS\b", re.I)):
        parent = tag.parent
        # 若文本本身就在一个单元格内（td/th），尝试获取相邻 td
        if parent.name in ["td", "th"]:
            # 找到同一行中下一个 td
            tr = parent.find_parent('tr')
            if tr:
                tds = tr.find_all(['td', 'th'])
                for i, td in enumerate(tds):
                    if td is parent and i + 1 < len(tds):
                        candidates.append(tds[i + 1])
                        break
        # 如果父元素是一个标题或强标签，可能紧随其后的是包含 concerns 的元素
        else:
            # 尝试寻找下一个兄弟元素
            sib = parent.find_next_sibling()
            if sib:
                candidates.append(sib)

    # 另外，查找表格 class 中带有 'ingredient-concerns' 的 table 或 section
    for tbl in soup.select("table.table-ingredient-concerns, section.ingredient-concerns, .ingredient-concerns-table-wrapper"):
        candidates.append(tbl)

    # 解析候选元素，按换行或 <br> 或 • 分割为多项
    for cand in candidates:
        # get_text with separator to preserve <br> boundaries
        text = cand.get_text(separator='||', strip=True)
        # split by our separator or by • bullet
        parts = []
        if '||' in text:
            parts = [p.strip() for p in text.split('||') if p.strip()]
        else:
            parts = [p.strip() for p in re.split(r'[\u2022•\n\r]+', text) if p.strip()]

        for part in parts:
            # skip short/heading-like parts like 'CONCERNS' or 'LEARN MORE'
            if len(part) < 3:
                continue
            if re.match(r"^CONCERNS?$", part.strip(), re.I):
                continue
            # extract level in parentheses at end
            m = re.search(r"\((high|moderate|low)\)\s*$", part, re.I)
            level = m.group(1).upper() if m else ""
            # remove leading bullet markers and trailing level parentheses
            concern = re.sub(r"^[\s\u2022•\-]+", "", part)
            concern = re.sub(r"\s*\((high|moderate|low)\)\s*$", "", concern, flags=re.I)
            concern = norm_text(concern)
            if concern:
                out.append({"level": level, "concern": concern})

    return out


def parse_ingredient_details_table(html: str) -> Dict[str, object]:
    """
    从包含成分详情的表格 HTML 中提取 Function 和 Concerns 列表（仅列表项，排除后面的说明段落）

    返回格式：{"function": "masking, ...", "concerns_list": ["Allergies/...", ...]}
    兼容两种常见结构：
      - concerns 项以 <li> 列表呈现
      - concerns 项以 • 符号或 <br> 换行呈现，后面跟随一段说明（该说明会被排除）
    """
    details = {"function": "", "concerns_list": []}
    if not html:
        return details

    soup = BeautifulSoup(html, "lxml")

    # 定位到 table/tbody（若 html 是片段，退化到整个片段）
    table = soup.find("table") or soup
    tbody = table.find("tbody") if table and table.find("tbody") else table

    # helper 查找 td 内容匹配某关键词的 td
    def find_td_by_label(label: str):
        return tbody.find("td", string=re.compile(rf"^{re.escape(label)}$", re.I))

    # 找 FUNCTION(S)
    func_td = find_td_by_label("FUNCTION(S)")
    if func_td:
        func_tr = func_td.find_parent("tr")
        if func_tr:
            tds = func_tr.find_all("td")
            if len(tds) > 1:
                details["function"] = norm_text(tds[1].get_text(" ", strip=True))

    # 找 CONCERNS
    concerns_label_td = tbody.find("td", string=re.compile(r"^CONCERNS?$", re.I))
    if concerns_label_td:
        concerns_tr = concerns_label_td.find_parent("tr")
        if concerns_tr:
            tds = concerns_tr.find_all("td")
            if len(tds) > 1:
                concerns_td = tds[1]

                # 优先用 <li>
                lis = concerns_td.select("li")
                if lis:
                    for li in lis:
                        txt = norm_text(li.get_text(" ", strip=True))
                        if txt:
                            details["concerns_list"].append(txt)
                    return details

                # 若没有 <li>，尝试在遇到第一个 <span>（通常为说明段）之前，只解析前面的内容
                span = concerns_td.find("span")
                if span:
                    parts_html = []
                    for child in concerns_td.contents:
                        if child is span:
                            break
                        parts_html.append(str(child))
                    fragment = BeautifulSoup("".join(parts_html), "lxml")
                    raw = fragment.get_text(separator="||", strip=True)
                else:
                    raw = concerns_td.get_text(separator="||", strip=True)

                # 按 ||、bullet、破折号（en/em dash）或换行分割（更稳健的分隔）
                # 注意：不要用 ASCII 连字符 '-' 作为分隔符（会把 "Non-reproductive" 拆开），
                # 但仍在后续步骤去除开头的 '-' 以兼容以 '-' 开头的 bullet 风格。
                parts = [p for p in re.split(r"\|\||[\u2022•\u2013\u2014]+|\n", raw) if p and p.strip()]
                # 过滤掉可能的标题/冗余词（例如 CONCERNS 本身），并去掉开头的 bullets/符号
                out_parts = []
                for p in parts:
                    p = p.strip()
                    if re.match(r"^CONCERNS?$", p, re.I):
                        continue
                    # 去掉开头的 bullet 符号或连字符
                    p = re.sub(r"^[\s\u2022•\-]+", "", p)
                    if p:
                        out_parts.append(norm_text(p))
                details["concerns_list"] = out_parts

    # 如果上面未命中 concerns（例如表头为 CONCERN 或表格以 th 为头），尝试表头索引法
    if not details.get("concerns_list"):
        try:
            for tbl in soup.find_all('table'):
                # 寻找表头中的 th 或第一行作为 header
                headers = []
                thead = tbl.find('thead')
                if thead:
                    headers = [norm_text(th.get_text(' ', strip=True)) for th in thead.find_all('th')]
                else:
                    # 兜底：检查第一行是否为 header（含 th 或第一行文本看起来像 header）
                    first_row = tbl.find('tr')
                    if first_row:
                        headers = [norm_text(c.get_text(' ', strip=True)) for c in first_row.find_all(['th','td'])]

                if not headers:
                    continue

                # 找到包含 CONCERN 的列索引（部分页面用单数 'CONCERN'）
                idx = None
                for i, h in enumerate(headers):
                    if re.search(r"\bCONCERN\b", h, re.I):
                        idx = i
                        break
                if idx is None:
                    continue

                # 收集该列在 tbody 中的所有单元格文本
                col_texts = []
                for tr in tbl.find_all('tr'):
                    tds = tr.find_all(['td','th'])
                    if len(tds) > idx:
                        txt = norm_text(tds[idx].get_text(' ', strip=True))
                        # 跳过 header 自身
                        if re.search(r"\bCONCERN\b", txt, re.I):
                            continue
                        if txt:
                            col_texts.append(txt)
                if col_texts:
                    details['concerns_list'] = col_texts
                    break
        except Exception:
            pass

    # 如果仍未找到 function，尝试更宽松地寻找页面中可能的 'Function' 标签或 'Function:' 文本
    if not (details.get('function') or '').strip():
        try:
            # 1) 在表格中寻找包含 FUNCTION 字样的 th/td
            found = False
            for tbl in soup.find_all('table'):
                for cell in tbl.find_all(['th','td']):
                    txt = norm_text(cell.get_text(' ', strip=True) or '')
                    if re.search(r"\bFUNCTIONS?\b", txt, re.I):
                        # 如果同一行有多个单元格，取下一单元格
                        tr = cell.find_parent('tr')
                        if tr:
                            cells = tr.find_all(['td','th'])
                            for i, c in enumerate(cells):
                                if c is cell and i + 1 < len(cells):
                                    val = norm_text(cells[i+1].get_text(' ', strip=True))
                                    if val:
                                        details['function'] = val
                                        found = True
                                        break
                        if found:
                            break
                if found:
                    break

            # 2) 回退：在整个片段的纯文本中查找 'Function:' 或 'FUNCTION(S):' 模式
            if not (details.get('function') or '').strip():
                alltext = soup.get_text(' ', strip=True)
                m = re.search(r"FUNCTION\(?S\)?\s*[:\-]\s*(.+?)(?:\.|$)", alltext, re.I)
                if m:
                    # 取到第一个句子/段落
                    details['function'] = norm_text(m.group(1))
        except Exception:
            pass

    return details


def split_by_plain_text(text: str) -> Dict[str, Dict[str, str]]:
    """
    仅有纯文本时的回退：用正则把各小标题作为分隔点切分
    """
    out = {"ingredients": {}, "directions": {}, "warnings": {}}
    if not text:
        return out
    T = text

    # 拼一个统一的大正则，把所有标题模式并起来，保留标题名以便定位
    def joined(pats: List[re.Pattern]) -> str:
        return "|".join([f"(?:{p.pattern})" for p in pats])

    big = re.compile(
        rf"(?P<ingredients>{joined(HEADINGS['ingredients'])})"
        rf"|(?P<directions>{joined(HEADINGS['directions'])})"
        rf"|(?P<warnings>{joined(HEADINGS['warnings'])})",
        re.I,
    )

    # 找到所有标题出现的位置
    hits = []
    for m in big.finditer(T):
        sec = None
        if m.group("ingredients"):
            sec = "ingredients"
        elif m.group("directions"):
            sec = "directions"
        elif m.group("warnings"):
            sec = "warnings"
        if sec:
            hits.append((sec, m.start(), m.end()))

    if not hits:
        return out

    # 以这些标题为锚，切分后续内容到下一个标题（只保留 text）
    for i, (sec, s, e) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(T)
        body = norm_text(T[e:end])
        out[sec] = {"text": body}

    return out


def parse_label_sections(label_html: str, label_text: str) -> Dict[str, Dict]:
    """
    先用 HTML 切；若拿不到，再用纯文本切
    最终统一补充 bullets[]
    """
    sections = split_by_html_headings(label_html)
    # 如果三个都空，再尝试纯文本
    if not any(sections[k] for k in sections):
        sections = split_by_plain_text(label_text)

    # 只保留纯文本 'text'（用户请求不需要任何 bullets）
    for k, v in sections.items():
        text = v.get("text", "")
        # 最终不要保留 html 字段；只输出纯文本 'text'
        sections[k] = {"text": text}
    return sections


def flatten_row(row: dict) -> dict:
    """
    生成便于看/导出的扁平字段
    """
    out = {
        "url": row.get("url"),
        "title": row.get("title"),
        "brand": row.get("brand"),
        "category": row.get("category"),
        # --- NEW: Ingredient List（来自采集阶段的 ingredient_list_text）
        "ingredients_from_packaging": (
            row.get("label_sections", {}).get("ingredients", {}).get("text", "")
        ),
        "directions_from_packaging": (
            row.get("label_sections", {}).get("directions", {}).get("text", "")
        ),
        "warnings_from_packaging": (
            row.get("label_sections", {}).get("warnings", {}).get("text", "")
        ),
    # --- NEW: 网站（右侧 WHERE TO BUY）
    "has_website_button": row.get("has_website_button"),
    "where_to_buy_urls": "; ".join(row.get("where_to_buy_urls") or []),
    "buy_button_urls": "; ".join(row.get("buy_button_urls") or []),
    # ingredient_concern: single string (deduped) e.g. 'MODERATE: Non-reproductive organ system toxicity; LOW: Ecotoxicology'
    # ingredient_concern: plain text summary (suitable for embedding)
    "ingredient_concern": row.get('ingredient_concern', ''),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", default="ewg_face_label_structured.jsonl")
    ap.add_argument("--out_csv", default="ewg_face_label_structured.csv")
    ap.add_argument("--add_ingdetails", action="store_true", help="Also parse per-ingredient tables and attach ingredient_details")
    ap.add_argument("--limit", type=int, default=0, help="Optional: only process first N records (0 = no limit)")
    args = ap.parse_args()

    structured_rows: List[dict] = []

    # 逐行读取输入 JSONL，逐条解析并生成记录
    processed = 0
    # session for optional ingredient page requests
    ING_SESSION = requests.Session()
    ING_SESSION.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; EWG-scraper/1.0; +https://example.com)'
    })
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.limit and processed >= args.limit:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # 跳过解析为 null 或 非 dict 的记录（防止 obj 为 None 导致 obj.get 抛错）
            if not isinstance(obj, dict):
                continue

            label_html = obj.get("label_information_html") or ""
            label_text = obj.get("label_information_text") or ""
            ingredient_concerns_html = obj.get("ingredient_concerns_html") or ""
            ingredient_details_html = obj.get("ingredient_details_html") or ""

            sections = parse_label_sections(label_html, label_text)

            # —— 每条记录都在循环内构建 —— #
            rec = {
                "url": obj.get("url"),
                "title": obj.get("title"),
                "brand": obj.get("brand"),

                # 结构化后的三段
                "label_sections": sections,

                # 原始 label（便于追溯）
                # "label_information_text": label_text,

                # 透传 Ingredient List & Where-to-buy（来自采集阶段）
                # 清空 where_to_buy_urls 以避免噪声，buy_button_urls 将被过滤
                "has_website_button": obj.get("has_website_button"),
                "buy_button_urls": [],
                # 我们不再输出 concerns 的原始列表（用户只要单个 concern 字段）
                # 在下面填充去重并合并后的字符串字段
                "ingredient_concern": "",
            }

            # 过滤并保留外部购买链接（如果存在）
            raw_buy = obj.get("buy_button_urls") or []
            if isinstance(raw_buy, str):
                raw_buy = [p.strip() for p in raw_buy.split(";") if p.strip()]

            def is_valid_purchase_link(u: str) -> bool:
                try:
                    from urllib.parse import urlparse
                    net = (urlparse(u).netloc or "").lower()
                    if not net:
                        return False
                    # 排除站内/非购买跳转域
                    blocked = ["ewg.org", "act.ewg.org", "play.google.com", "apps.apple.com"]
                    if any(b in net for b in blocked):
                        return False
                    return True
                except Exception:
                    return False

            filtered = [u for u in raw_buy if is_valid_purchase_link(u)]
            rec["buy_button_urls"] = filtered
            # 解析 ingredient concerns HTML（若存在），并只保留 level 非空的项
            parsed_ic = parse_ingredient_concerns_list(ingredient_concerns_html)
            parsed_ic = [it for it in parsed_ic if (it.get('level') or '').strip()]
            # 按 level 分组为 mapping，并保持原始出现顺序，同时去重每个 level 下的 concern
            mapping = {}
            seen = set()
            for it in parsed_ic:
                lvl = (it.get('level') or '').strip().lower()
                conc = (it.get('concern') or '').strip()
                if not lvl or not conc:
                    continue
                key = (lvl, conc)
                if key in seen:
                    continue
                seen.add(key)
                mapping.setdefault(lvl, [])
                mapping[lvl].append(conc)

            # Ensure mapping contains only levels present (no empty lists)
            final_map = {k: v for k, v in mapping.items() if v}
            # 主要输出：把 concerns 作为纯文本（便于 embedding）
            parts = []
            for lvl in ["high", "moderate", "low"]:
                items = final_map.get(lvl)
                if items:
                    parts.append(f"{lvl.upper()}: {', '.join(items)}")
            text_summary = '; '.join(parts)
            rec['ingredient_concern'] = text_summary
            # 备份原始 mapping 到单独字段（仅 JSONL 使用，便于追溯）
            rec['ingredient_concern_map'] = final_map
            # 透传 category（若在爬虫中采集到了）
            if obj.get('category'):
                rec['category'] = obj.get('category')
            # optionally add per-ingredient details parsed from tables
            if args.add_ingdetails:
                ing_details = []
                html_candidates = []
                # prefer a dedicated field if present, otherwise fall back to common fields
                if obj.get('ingredient_details_html'):
                    html_candidates.append(obj.get('ingredient_details_html'))
                if obj.get('label_information_html'):
                    html_candidates.append(obj.get('label_information_html'))
                if obj.get('ingredient_concerns_html'):
                    html_candidates.append(obj.get('ingredient_concerns_html'))
                # If crawler saved ingredient_pages (url + html), parse them first
                if obj.get('ingredient_pages'):
                    try:
                        for page in obj.get('ingredient_pages'):
                            page_html = page.get('html') or ''
                            page_url = page.get('url') or ''
                            if not page_html:
                                continue
                            soup2 = BeautifulSoup(page_html, 'lxml')
                            found_table = False
                            for t2 in soup2.select('table'):
                                res2 = parse_ingredient_details_table(str(t2))
                                # record ingredient name from URL if available
                                if page_url and not res2.get('ingredient_name'):
                                    # try to get last path segment as name fallback
                                    try:
                                        name = page_url.rstrip('/').split('/')[-1]
                                        res2['ingredient_name'] = name
                                    except Exception:
                                        res2['ingredient_name'] = ''
                                if (res2.get('function') or '').strip() or res2.get('concerns_list'):
                                    ing_details.append(res2)
                                    found_table = True
                            if not found_table:
                                cand = soup2.find(string=re.compile(r"FUNCTION\(?S\)?", re.I))
                                if cand:
                                    parent = cand.parent
                                    txt = parent.get_text(" ", strip=True)
                                    m = re.search(r"FUNCTION\(?S\)?[:\-]?\s*(.*)$", txt, re.I)
                                    if m:
                                        ing_details.append({
                                            'function': norm_text(m.group(1)),
                                            'concerns_list': [],
                                            'ingredient_name': page_url,
                                        })
                    except Exception:
                        pass
                for html in html_candidates:
                    if not html:
                        continue
                    soup = BeautifulSoup(html, 'lxml')
                    for table in soup.select('table'):
                        res = parse_ingredient_details_table(str(table))
                        # 如果没有 function，尝试在 table 或其附近寻找 ingredient 链接并抓取 ingredient 页面
                        if not (res.get('function') or '').strip():
                            # 优先在 table 内查找链接
                            a = None
                            try:
                                a = table.select_one("a[href*='/skindeep/ingredients/']")
                            except Exception:
                                a = None
                            if not a:
                                # 尝试查找前后相邻的 a 标签（有些页面把 link 放在 table 上方或下方）
                                try:
                                    a = table.find_previous('a', href=re.compile(r"/skindeep/ingredients/"))
                                except Exception:
                                    a = None
                            if not a:
                                try:
                                    a = table.find_next('a', href=re.compile(r"/skindeep/ingredients/"))
                                except Exception:
                                    a = None

                            if a and a.get('href'):
                                href = a.get('href')
                                ing_url = urljoin('https://www.ewg.org', href)
                                try:
                                    # be polite and not hammer the site
                                    time.sleep(0.3)
                                    r = ING_SESSION.get(ing_url, timeout=15)
                                    if r.status_code == 200 and r.text:
                                        # parse ingredient page's table(s) and try to extract function
                                        soup2 = BeautifulSoup(r.text, 'lxml')
                                        # look for a table with FUNCTION(S) or for the standard details table
                                        found_func = False
                                        for t2 in soup2.select('table'):
                                            res2 = parse_ingredient_details_table(str(t2))
                                            if (res2.get('function') or '').strip():
                                                res['function'] = res2.get('function')
                                                # merge concerns_list if absent
                                                if not res.get('concerns_list') and res2.get('concerns_list'):
                                                    res['concerns_list'] = res2.get('concerns_list')
                                                found_func = True
                                                break
                                        # 若未找到 table，可尝试在页面中直接定位 FUNCTION(s) 文本块 (fallback)
                                        if not found_func:
                                            # 搜索包含 'FUNCTION' 文本的行/段落
                                            cand = soup2.find(string=re.compile(r"FUNCTION\(?S\)?", re.I))
                                            if cand:
                                                parent = cand.parent
                                                txt = parent.get_text(" ", strip=True)
                                                # 取 ':' 或 'FUNCTION(S)' 后面的部分
                                                m = re.search(r"FUNCTION\(?S\)?[:\-]?\s*(.*)$", txt, re.I)
                                                if m:
                                                    res['function'] = norm_text(m.group(1))
                                except Exception:
                                    # 若抓取失败，忽略并继续
                                    pass

                        if (res.get('function') or '').strip() or res.get('concerns_list'):
                            ing_details.append(res)
                    # 回退：如果当前 html 片段没有 table-based details，或我们仍想全面覆盖，
                    # 尝试从 HTML 中提取所有 /skindeep/ingredients/ 链接并抓取 ingredient 页面
                    links = []
                    try:
                        for a in soup.select("a[href*='/skindeep/ingredients/']"):
                            href = a.get('href')
                            if not href:
                                continue
                            full = urljoin('https://www.ewg.org', href)
                            links.append((full, a.get_text(" ", strip=True)))
                    except Exception:
                        links = []
                    # 去重并遍历
                    seen_links = set()
                    for full, link_text in links:
                        if full in seen_links:
                            continue
                        seen_links.add(full)
                        try:
                            time.sleep(0.25)
                            r = ING_SESSION.get(full, timeout=12)
                            if r.status_code != 200 or not r.text:
                                continue
                            soup2 = BeautifulSoup(r.text, 'lxml')
                            # parse tables on ingredient page
                            found_table = False
                            for t2 in soup2.select('table'):
                                res2 = parse_ingredient_details_table(str(t2))
                                # include the ingredient name from link text if available
                                if link_text and not res2.get('ingredient_name'):
                                    res2['ingredient_name'] = link_text
                                if (res2.get('function') or '').strip() or res2.get('concerns_list'):
                                    ing_details.append(res2)
                                    found_table = True
                                    break
                            # small heuristic: also try to find function text blocks if no tables found
                            if not found_table:
                                cand = soup2.find(string=re.compile(r"FUNCTION\(?S\)?", re.I))
                                if cand:
                                    parent = cand.parent
                                    txt = parent.get_text(" ", strip=True)
                                    m = re.search(r"FUNCTION\(?S\)?[:\-]?\s*(.*)$", txt, re.I)
                                    if m:
                                        ing_details.append({
                                            'function': norm_text(m.group(1)),
                                            'concerns_list': [],
                                            'ingredient_name': link_text,
                                        })
                        except Exception:
                            continue
                rec['ingredient_details'] = ing_details
            structured_rows.append(rec)
            processed += 1

    # 输出结构化 JSONL（完整信息）
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in structured_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 扁平 CSV（列清晰便于查看）
    flat = [flatten_row(r) for r in structured_rows]
    pd.DataFrame(flat).to_csv(args.out_csv, index=False, encoding="utf-8")

    print(f"Done: {args.out_jsonl} / {args.out_csv}")


if __name__ == "__main__":
    main()
