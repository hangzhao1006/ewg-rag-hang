#!/usr/bin/env python3
"""
将结构化的 JSONL（如 ewg_face_label_structured.jsonl）导出为一组 .txt 文件，便于后续建语料或手动检查。

示例：
python scripts/jsonl_to_txt.py --in_jsonl ../ewg_face_label_structured.jsonl --out_dir ../input-datasets/books --limit 200

功能：
- 根据 title 生成安全的文件名（前缀为四位行号）
- 支持 --dry-run（仅列出将要写入/覆盖的文件）、--force（覆盖已存在文件）
"""

import argparse
import json
import os
import re
from typing import List


def safe_fname(index: int, title: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|\n]+', ' ', (title or 'product'))[:80].strip()
    return f"{index:04d}-{safe}.txt"


def record_text(rec: dict) -> str:
    parts: List[str] = []
    if rec.get('title'):
        parts.append(rec.get('title'))
    if rec.get('brand'):
        parts.append('Brand: ' + rec.get('brand'))
    if rec.get('category'):
        parts.append('Category: ' + rec.get('category'))
    ls = rec.get('label_sections', {}) or {}
    for k in ['ingredients', 'directions', 'warnings']:
        v = ls.get(k, {}).get('text') if ls.get(k) else None
        if v:
            parts.append(k.capitalize() + ': ' + v)
    if rec.get('ingredient_concern'):
        parts.append('Ingredient concerns: ' + rec.get('ingredient_concern'))
    return '\n\n'.join(parts).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='Overwrite existing files')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    written = 0
    skipped_empty = 0
    overwritten = 0
    to_write = []

    with open(args.in_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.limit and total >= args.limit:
                break
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = record_text(obj)
            fname = safe_fname(i, obj.get('title'))
            outpath = os.path.join(args.out_dir, fname)
            if not text:
                skipped_empty += 1
                continue
            to_write.append((outpath, text))

    # Dry run: just print summary
    if args.dry_run:
        print(f"Total records inspected: {total}")
        print(f"Would write files: {len(to_write)} (skipped empty: {skipped_empty})")
        # show a sample of first 30
        for p, _ in to_write[:30]:
            exists = os.path.exists(p)
            print(f"{'EXISTS' if exists else 'NEW   '}: {p}")
        return

    # Actual write
    for p, txt in to_write:
        if os.path.exists(p) and not args.force:
            # skip unless force
            continue
        if os.path.exists(p) and args.force:
            overwritten += 1
        with open(p, 'w', encoding='utf-8') as out:
            out.write(txt)
        written += 1

    print(f"Total inspected: {total}")
    print(f"Written: {written} (overwritten: {overwritten})")
    print(f"Skipped-empty: {skipped_empty}")


if __name__ == '__main__':
    main()
