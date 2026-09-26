#!/usr/bin/env python3
"""PubTables-QA scorer: type-aware accuracy (TAA) and ANLS.

Self-contained (standard library only). Scores model predictions against
data/test.jsonl of the PubTables-QA release.

Pipeline for every question
  1. Rule-based answer extraction from the raw response (`extract_answer`).
  2. TAA: the gold answer is assigned an answer type from the question and gold
     text only (`answer_spec`; reviewed types for a small set of questions are
     embedded in SPEC_OVERRIDES). The extracted answer is judged correct by
     type-specific rules (`taa_correct`): numbers with units and the rounding the
     question asks for, intervals, page lists, table IDs, table/section + page
     pairs, ordered tuples and unordered sets, and conservative text matching.
     Reviewed alternative surface forms (EQUIVALENCES) are applied to every
     model alike. No similarity threshold ever grants correctness.
  3. ANLS (tau = 0.5) on the same extracted answer, after domain normalization.
  4. Optional sentence post-processing: if a prediction record also carries a
     short answer extracted by a gold-blind model (--extracted-key), it is used
     only when every part of it occurs verbatim in the raw response, and the
     question counts as correct if either the original or the extracted answer
     is correct (ANLS takes the higher value).

Usage
  python score_taa.py --data data/test.jsonl --pred predictions.jsonl
  python score_taa.py --data data/test.jsonl --pred predictions.jsonl \\
      --pred-key response --extracted-key extracted --out item_scores.jsonl

Prediction file: JSON Lines with a `qid` and the raw model response (the key is
auto-detected from pred_answer / prediction / response / answer / pred unless
--pred-key is given), or a JSON object {qid: response}.
Questions without a prediction count as wrong; the number is reported.
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, ROUND_HALF_UP
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# 1. Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_ANSWER_IS_RE = re.compile(r"(?:the\s+)?answer\s+is[:\s]+(.+)", re.IGNORECASE)
_ANSWER_COLON_RE = re.compile(r"^Answer:\s*(.+)", re.MULTILINE)
_FINAL_ANSWER_RE = re.compile(r"(?:final\s+answer)[:\s]+(.+)", re.IGNORECASE)
_THEREFORE_RE = re.compile(r"(?:therefore|thus|hence)[,:\s]+(?:the\s+(?:answer|value|result|total|sum|difference|ratio|count|number)\s+is\s+)?(.+)", re.IGNORECASE)
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_REPEATED_ZERO_RE = re.compile(r"^0\.0{5,}0*$")
_STRIP_MARKDOWN_RE = re.compile(r"[*_`#>]")


def _strip_markdown(s):
    s = _STRIP_MARKDOWN_RE.sub("", s)
    s = re.sub(r"^\s*[-•]\s*", "", s)
    s = re.sub(r"^\s*\d+\.\s+", "", s)
    return s.strip()


def _clean_extracted(s):
    s = s.strip().rstrip(".,;:")
    s = _strip_markdown(s)
    s = s.strip("\"'")
    return s.strip()


def extract_answer(pred):
    pred = pred.strip()
    if not pred:
        return pred
    if len(pred) <= 50 and "\n" not in pred:
        if _REPEATED_ZERO_RE.match(pred):
            return "0"
        return pred
    if _REPEATED_ZERO_RE.match(pred.split("\n")[0].strip()):
        return "0"
    fw = pred.split(",")[0].split(".")[0].strip().lower()
    if fw in ("yes", "no", "true", "false"):
        return fw.capitalize() if fw in ("yes", "no") else fw
    m = _BOXED_RE.search(pred)
    if m:
        return m.group(1).replace("\\%", "%").strip()
    m = _FINAL_ANSWER_RE.search(pred)
    if m:
        v = _clean_extracted(m.group(1).split("\n")[0])
        if v and v.lower() not in ("", "not answerable"):
            return v
    matches = list(_ANSWER_IS_RE.finditer(pred))
    if matches:
        v = _clean_extracted(matches[-1].group(1).split("\n")[0])
        if v and v.lower() not in ("", "not answerable"):
            return v
    matches_colon = list(_ANSWER_COLON_RE.finditer(pred))
    if matches_colon:
        v = _clean_extracted(matches_colon[-1].group(1).split("\n")[0])
        if v and v.lower() not in ("", "not answerable"):
            return v
    lines = [l.strip() for l in pred.split("\n") if l.strip()]
    if len(lines) > 1:
        last = lines[-1]
        m = _THEREFORE_RE.match(last)
        if m:
            v = _clean_extracted(m.group(1))
            if v:
                return v
        bolds_last = _BOLD_RE.findall(last)
        rest_text = _BOLD_RE.sub("", last).strip()
        if bolds_last and len(rest_text) < 20:
            return bolds_last[-1].strip()
    bolds = _BOLD_RE.findall(pred)
    if bolds:
        lb = bolds[-1].strip()
        lbp = pred.rfind(f"**{lb}**")
        ab = pred[lbp + len(lb) + 4:].strip()
        if len(ab) < 30 and lbp > len(pred) * 0.5:
            return _clean_extracted(lb)
    if len(lines) > 1:
        first = lines[0]
        if len(first) <= 30 and len(pred) > 100:
            fc = _strip_markdown(first)
            if fc and not fc.endswith(":"):
                return fc
    if len(lines) > 1:
        last = lines[-1]
        if len(last) <= 60 and len(pred) > 100:
            mt = _THEREFORE_RE.search(last)
            if mt:
                return _clean_extracted(mt.group(1))
            lc = _strip_markdown(last)
            if lc and not lc.endswith(":"):
                nums = re.findall(r"-?\d+(?:\.\d+)?", lc)
                if nums and len(lc) < 25:
                    return lc
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANLS with domain normalization
# ─────────────────────────────────────────────────────────────────────────────

def _edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if s1[i - 1] == s2[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def anls_single(gold, pred, tau=0.5):
    g, p = gold.strip().lower(), pred.strip().lower()
    mx = max(len(g), len(p))
    if mx == 0:
        return 1.0
    nl = _edit_distance(g, p) / mx
    return 0.0 if nl >= tau else 1.0 - nl


_BOOL_MAP = {"yes": "true", "no": "false", "correct": "true", "incorrect": "false",
             "true": "true", "false": "false", "1": "true", "0": "false"}
_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
              "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
              "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
              "seventy": 70, "eighty": 80, "ninety": 90}


def _normalize_basic(s):
    s = str(s).strip().lower().replace("\n", " ")
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[,\.;:]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_bool(s):
    c = str(s).strip().lower().rstrip(".,;:")
    if c in _BOOL_MAP:
        return _BOOL_MAP[c]
    f = re.split(r"[\s,;.]", c)[0].rstrip(".,;:")
    return _BOOL_MAP.get(f, c)


def _is_bool(s):
    return str(s).strip().lower().rstrip(".,;:") in _BOOL_MAP


def _try_numeric(s):
    s = s.strip().rstrip("%")
    try:
        v = float(s)
        if v == int(v):
            return str(int(v))
        return f"{v:g}"
    except (ValueError, OverflowError):
        return None


def _words_to_int(s):
    """Leading English number word (0-99, e.g. 'Three rows.', 'twenty-one') as an int."""
    t = str(s).strip().lower()
    if re.search(r"\d", t):
        return None
    toks = re.findall(r"[a-z]+", t.replace("-", " "))
    if not toks or toks[0] not in _NUM_WORDS:
        return None
    v = _NUM_WORDS[toks[0]]
    if v >= 20 and len(toks) > 1 and toks[1] in _NUM_WORDS and _NUM_WORDS[toks[1]] < 10:
        v += _NUM_WORDS[toks[1]]
    return v


def domain_normalize(gold, pred):
    gn, pn = _normalize_basic(gold), _normalize_basic(pred)
    if _try_numeric(str(gold).strip()) is not None and not _is_bool(gn):
        w = _words_to_int(pred)
        if w is not None:
            pred = str(w)
            pn = _normalize_basic(pred)
    if _is_bool(gn):
        return _norm_bool(gn), _norm_bool(pn)
    if re.search(r"\bpage\s+\d+\b", str(gold).lower()):
        gn2 = " ".join(sorted(re.findall(r"\d+", str(gold)), key=int))
        pn2 = " ".join(sorted(re.findall(r"\d+", str(pred)), key=int))
        return gn2, pn2
    gnum = _try_numeric(str(gold).strip())
    pnum = _try_numeric(str(pred).strip())
    if gnum is not None and pnum is not None:
        return gnum, pnum
    if gnum is not None and pnum is None:
        m = re.search(r'=\s*(-?\d+\.?\d*)\s*$', str(pred).strip())
        if m:
            pnum2 = _try_numeric(m.group(1))
            if pnum2 is not None:
                return gnum, pnum2
        m = re.match(r'(-?\d+\.?\d*)\s*(?:%|days?|weeks?|years?|months?|patients?|ml|mg|kg|hours?|h|/[MF])\s*$',
                     str(pred).strip(), re.IGNORECASE)
        if m:
            pnum2 = _try_numeric(m.group(1))
            if pnum2 is not None:
                return gnum, pnum2
    return gn, pn


def anls(gold, extracted):
    gn, pn = domain_normalize(str(gold), extracted)
    return anls_single(gn, pn)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Type-aware accuracy: parsing helpers and strict type rules
# ─────────────────────────────────────────────────────────────────────────────

NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:e[+-]?\d+)?"
WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
WORDS.update(dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10))))
UNIT_ALIASES = {
    '%': '%', 'percent': '%', 'per cent': '%', 'percentage points': 'percentage points',
    'percentage point': 'percentage points', '°c': '°c', '°f': '°f',
    'min': 'min', 'mins': 'min', 'minute': 'min', 'minutes': 'min',
    'h': 'hours', 'hr': 'hours', 'hrs': 'hours', 'hour': 'hours', 'hours': 'hours',
    'day': 'days', 'days': 'days', 'week': 'weeks', 'weeks': 'weeks',
    'month': 'months', 'months': 'months', 'year': 'years', 'years': 'years', 'yrs': 'years',
}
for _unit in ['kg', 'g', 'mg', 'μg', 'ng', 'ml', 'l', 'dl', 'mm', 'cm', 'm', 'km',
              'iu/l', 'u/l', 'ku/l', 'da/aa', 'mg/dl', 'mmol/l', 'μmol/l']:
    UNIT_ALIASES[_unit] = _unit
COUNT_UNITS = {'row', 'column', 'patient', 'participant', 'subject', 'case', 'sample',
               'cell', 'table', 'page', 'item', 'marker', 'entry', 'study', 'phage product',
               'variable', 'gene', 'category', 'part', 'site', 'measure', 'domain', 'theme',
               'feature', 'level', 'criterion', 'outcome', 'value'}
for _unit in COUNT_UNITS:
    UNIT_ALIASES[_unit] = _unit
    UNIT_ALIASES[_unit + 's'] = _unit
UNIT_ALIASES.update({'studies': 'study', 'entries': 'entry', 'marker entries': 'entry',
                     'categories': 'category', 'criteria': 'criterion'})


def surface(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = s.translate(str.maketrans({'−': '-', '–': '-', '—': '-', '‐': '-', '‑': '-',
                                   '’': "'", '‘': "'", '“': '"', '”': '"', 'µ': 'μ'}))
    return re.sub(r'\s+', ' ', s).strip()


def clean(s):
    s = surface(s)
    # Strip only outer wrappers: never delete interior decimal points, minus signs or IDs.
    s = s.strip('`" ')
    if s.startswith('**') and s.endswith('**'):
        s = s[2:-2].strip()
    s = re.sub(r'^(?:final\s+answer|answer)\s*:\s*', '', s)
    return s.rstrip('.,; ').strip('" ')


def text_key(s):
    s = clean(s)
    return re.sub(r'\s*([,;:()\[\]])\s*', r'\1', s)


def word_number(s):
    toks = surface(s).replace('-', ' ').split()
    if not toks or any(t not in WORDS and t not in ('hundred', 'thousand', 'and') for t in toks):
        return None
    # Full-string grammar: "three or four" or "one hundred" are not accepted as one number.
    if len(toks) == 1 and toks[0] in WORDS:
        return Decimal(WORDS[toks[0]])
    if len(toks) == 2 and WORDS.get(toks[0], 0) >= 20 and 0 < WORDS.get(toks[1], 100) < 10:
        return Decimal(WORDS[toks[0]] + WORDS[toks[1]])
    return None


def numeric(s):
    s = clean(s).replace('\\%', '%')
    s = re.sub(r'^p\s*=\s*', '', s)
    m = re.fullmatch(r'(' + NUM + r')\s*(.*?)', s)
    if m:
        suffix = m[2].strip()
        if suffix and suffix not in UNIT_ALIASES:
            return None
        try:
            return Decimal(m[1].replace(',', '')), UNIT_ALIASES.get(suffix, '')
        except InvalidOperation:
            return None
    w = word_number(s)
    if w is not None:
        return w, ''
    for suffix in sorted(UNIT_ALIASES, key=len, reverse=True):
        if s.endswith(' ' + suffix):
            w = word_number(s[:-len(suffix)].strip())
            if w is not None:
                return w, UNIT_ALIASES[suffix]
    return None


def precision(question):
    """Decimal places explicitly requested by the question ("rounded to two decimal places")."""
    q = surface(question)
    words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
             'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6}
    m = re.search(r'(?:round(?:ed)?(?:\s+\w+){0,3}\s+to|calculate\s+to)\s+(?:the\s+)?(\d+|one|two|three|four|five|six|first|second|third|fourth|fifth|sixth)\s+decimal', q)
    return int(m[1]) if m and m[1].isdigit() else words.get(m[1]) if m else None


def numeric_equal(g, p, spec):
    ga, pa = numeric(g), numeric(p)
    if pa is None:
        m = re.search(r'=\s*(' + NUM + r')\s*([^=\n]*)$', clean(p))
        if m:
            pa = numeric(m[1] + ' ' + m[2])
    if ga is None or pa is None:
        return False
    gv, gu = ga
    pv, pu = pa
    if pu and gu and pu != gu:
        return False
    if pu and not gu:
        q = surface(spec.get('question', ''))
        allowed = {canonical for alias, canonical in UNIT_ALIASES.items()
                   if re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', q)}
        physical = allowed - COUNT_UNITS
        count_question = bool(re.search(r'\bhow many\b|\bnumber of\b|\bcount\b', q))
        if (physical and pu not in physical) or (count_question and not physical and pu not in COUNT_UNITS):
            return False
    dp = spec.get('decimals')
    if dp is not None:
        quantum = Decimal(1).scaleb(-dp)
        try:
            return gv.quantize(quantum, rounding=ROUND_HALF_UP) == pv.quantize(quantum, rounding=ROUND_HALF_UP)
        except InvalidOperation:
            return False
    return gv == pv


def parts(s, split_and=False):
    """Split top-level list delimiters, keeping parentheses and quoted entities intact."""
    s = str(s).replace('“', '"').replace('”', '"')
    result, start, depth, quoted = [], 0, 0, False
    for i, c in enumerate(s):
        if c == '"':
            quoted = not quoted
        if not quoted:
            if c in '([':
                depth += 1
            elif c in ')]':
                depth = max(0, depth - 1)
            elif depth == 0 and (c in ';\n' or (c == ',' and not (i and i + 1 < len(s) and s[i - 1].isdigit() and s[i + 1].isdigit()))):
                result.append(s[start:i].strip())
                start = i + 1
    result.append(s[start:].strip())
    result = [x for x in result if x]
    if split_and and len(result) == 1:
        result = re.split(r'\s+and\s+', result[0], flags=re.I)
    return [re.sub(r'^(?:and\s+|[-•]\s+)', '', x, flags=re.I).strip() for x in result]


def page_list(s, ordered=False):
    s = clean(s)
    s = re.sub(r'\b(?:pages?|pp?\.)\s*', '', s)
    for w, n in WORDS.items():
        s = re.sub(r'\b' + w + r'\b', str(n), s)
    s = re.sub(r'\b(?:and|to)\b', lambda m: ',' if m[0] == 'and' else '-', s)
    s = s.replace('→', '-')
    if not re.fullmatch(r'\d+(?:\s*(?:[,;/-]|\s)\s*\d+)*', s):
        return None
    seq = []
    for found in re.finditer(r'\d+\s*-\s*\d+|\d+', s):
        token = found[0]
        if re.fullmatch(r'\d+\s*-\s*\d+', token):
            a, b = map(int, re.split(r'\s*-\s*', token))
            if b < a or b - a > 1000:
                return None
            seq.extend(range(a, b + 1))
        elif token.isdigit():
            seq.append(int(token))
        else:
            return None
    return tuple(seq) if ordered else frozenset(seq)


def table_ids(s):
    s = clean(s)
    s = re.sub(r'\btables?\s*', '', s)
    s = re.sub(r'\band\b', ',', s)
    s = re.sub(r',\s*,', ',', s)
    tokens = [x.strip() for x in re.split(r'[,;]', s) if x.strip()]
    if not tokens or not all(re.fullmatch(r'[a-z]?\d+[a-z]?', x) for x in tokens):
        return None
    return tuple(tokens)


def interval(s):
    s = clean(s)
    m = re.fullmatch(r'(' + NUM + r')\s*(%?)\s*(?:-|to)\s*(' + NUM + r')\s*(.*)', s)
    if not m:
        return None
    unit = m[4].strip() or m[2]
    a, b = numeric(m[1] + ' ' + unit), numeric(m[3] + ' ' + unit)
    return (m[1] + ' ' + unit, m[3] + ' ' + unit) if a and b else None


def strict_table_page(s):
    s = clean(s)
    m = re.fullmatch(r'(table\s+[a-z]?\d+[a-z]?)\s*(?:on\s+|[,;]\s*)?(pages?\s*\d.*)', s)
    if not m:
        return None
    return table_ids(m[1]), page_list(m[2])


def section_page(s):
    s = clean(s)
    m = re.fullmatch(r'(.+?)\s*(?:[,;]\s*|\s+on\s+)(?:pages?\s*)?(\d[\d\s,;\-]*(?:and\s+\d+)?)', s)
    if not m:
        return None
    section = re.sub(r'^(?:the\s+)?', '', m[1])
    section = re.sub(r'\s+section$', '', section)
    return text_key(section), page_list(m[2])


def strict_component_equal(g, p, spec):
    if spec.get('component_mode') == 'text':
        return text_key(g) == text_key(p)
    if clean(g) in ('yes', 'true', 'correct', 'no', 'false', 'incorrect'):
        aliases = {'yes': 1, 'true': 1, 'correct': 1, 'no': 0, 'false': 0, 'incorrect': 0}
        return clean(p) in aliases and aliases[clean(g)] == aliases[clean(p)]
    if numeric(g) is not None:
        return numeric_equal(g, p, spec)
    gi, pi = interval(g), interval(p)
    if gi:
        return bool(pi and all(numeric_equal(a, b, spec) for a, b in zip(gi, pi)))
    return text_key(g) == text_key(p)


def classify(qid, question, gold):
    """Answer type from the question and gold only (frozen before predictions are read)."""
    g, q = gold, question
    spec = {'qid': qid, 'question': q, 'gold': g, 'decimals': precision(q), 'type': 'text_exact'}
    if clean(g) in ('yes', 'no', 'true', 'false', 'correct', 'incorrect'):
        spec['type'] = 'boolean'
    elif strict_table_page(g):
        spec['type'] = 'table_page'
    elif re.match(r'^tables?\s', clean(g)) and table_ids(g):
        spec['type'] = 'table_set'
    elif page_list(g) is not None and (re.match(r'^pages?\s*\d', clean(g)) or re.search(r'(?:which|what)\s+(?:two\s+)?pages?|(?:between|across)\s+which\s+pages|from which page|union of page', q, re.I)):
        spec['type'] = 'page_set'
    elif numeric(g) is not None:
        spec['type'] = 'numeric'
    elif interval(g):
        spec['type'] = 'interval'
    else:
        ps = parts(g, split_and=True)
        if len(ps) > 1 and all(numeric(p) is not None for p in ps):
            spec.update(type='ordered_tuple', components=ps, split_and=True)
    return spec


def strict_score(spec, pred):
    typ, g = spec['type'], spec['gold']
    p = str(pred).strip()
    if not p:
        return False
    if typ == 'numeric':
        return numeric_equal(g, p, spec)
    if typ == 'boolean':
        boolean = {'yes': True, 'true': True, 'correct': True, '1': True,
                   'no': False, 'false': False, 'incorrect': False, '0': False}
        return clean(p) in boolean and boolean[clean(p)] == boolean[clean(g)]
    if typ == 'interval':
        gi, pi = interval(g), interval(p)
        return bool(pi and all(numeric_equal(a, b, spec) for a, b in zip(gi, pi)))
    if typ in ('page_set', 'page_tuple'):
        gg, pp = page_list(g, typ == 'page_tuple'), page_list(p, typ == 'page_tuple')
        return pp is not None and gg == pp
    if typ in ('table_set', 'table_tuple'):
        gg, pp = table_ids(g), table_ids(p)
        return pp is not None and (set(gg) == set(pp) if typ == 'table_set' else gg == pp)
    if typ in ('table_page', 'section_page'):
        parser = strict_table_page if typ == 'table_page' else section_page
        gg, pp = parser(g), parser(p)
        return pp is not None and pp[1] is not None and gg == pp
    if typ in ('unordered_set', 'ordered_tuple'):
        gg = spec.get('components') or parts(g, spec.get('split_and', False))
        pp = ([x.strip() for x in p.split(spec['delimiter'])] if spec.get('delimiter')
              else parts(p, spec.get('split_and', False)))
        if len(pp) == 1 and len(gg) > 1 and all(numeric(x) is not None for x in gg):
            pp = re.split(r'\s*[,;]\s*|\s+and\s+', p, flags=re.I)
        if typ == 'ordered_tuple':
            return len(gg) == len(pp) and all(strict_component_equal(a, b, spec) for a, b in zip(gg, pp))
        return {text_key(x) for x in gg} == {text_key(x) for x in pp}
    return text_key(g) == text_key(p)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Type-aware accuracy: equivalent formats and actual rounding
# ─────────────────────────────────────────────────────────────────────────────

def lexical(s):
    """Formatting equivalence only; numbers, signs, operators and letters are kept."""
    s = surface(s).strip('`"\' .!?;')
    s = re.sub(r'^(?:the|a|an)\s+', '', s)
    s = re.sub(r'\borganisation\b', 'organization', s)
    s = re.sub(r'\bin-patient\b', 'inpatient', s)
    s = re.sub(r'(?<=[a-z])\s*-\s*(?=[a-z])', '', s)
    s = re.sub(r'\s+-\s+', ' ', s)
    s = re.sub(r'(?<!\d)\.|\.(?!\d)', '', s)
    # '/' (units), '*' (significance), '<', '>' and '-' (signed values, ranges) are kept.
    return re.sub(r'[\s,;:()\[\]{}\"\'`_|!]+', '', s)


def numeric_signature(s):
    t = surface(s)
    t = re.sub(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', lambda m: m[0].replace(',', ''), t)
    return tuple(re.findall(r'\d+(?:\.\d+)?', t))


def text_equal(g, p):
    # Literature search strings need their grouping and wildcards.
    if re.search(r'\badj\d|\.ti,|therapist/', str(g), re.I):
        return text_key(g) == text_key(p)
    return numeric_signature(g) == numeric_signature(p) and lexical(g) == lexical(p)


def numeric_prediction(s):
    val = numeric(s)
    if val is not None:
        return val
    t = clean(s)
    m = re.search(r'=\s*(' + NUM + r')\s*([^=\n]*)$', t)
    if m and numeric(m[1] + ' ' + m[2]) is not None:
        return numeric(m[1] + ' ' + m[2])
    lines = [numeric(x) for x in str(s).splitlines() if x.strip()]
    if len(lines) > 1 and all(x is not None and x == lines[0] for x in lines):
        return lines[0]
    return None


def units_equal(gu, pu, question):
    if not pu:
        return True
    if gu:
        return pu == gu
    q = surface(question)
    if re.search(r'\bwhat percentage\b|\bwhat percent\b', q):
        return pu == '%'
    if re.search(r'\bhow many\b|\bcount (?:exactly|the|all)\b', q):
        return pu in COUNT_UNITS
    return True  # no explicit gold unit: a recognised suffix is a display convention


def rounded_equal(a, b, decimals=None):
    if a == b:
        return True
    if a.is_zero() or b.is_zero() or a.is_signed() != b.is_signed():
        return False
    if decimals is not None:
        q = Decimal(1).scaleb(-decimals)
        return any(a.quantize(q, rounding=mode) == b.quantize(q, rounding=mode)
                   for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN))
    if a == a.to_integral():
        return False  # an integer gold (count/ID) gets no rounding allowance
    ea, eb = a.normalize().as_tuple().exponent, b.normalize().as_tuple().exponent
    if ea == eb:
        return False
    coarse, fine = (a, b) if ea > eb else (b, a)
    # A coarser prediction needs at least two significant digits.
    if coarse == b and len(coarse.normalize().as_tuple().digits) < 2:
        return False
    q = Decimal(1).scaleb(max(ea, eb))
    try:
        return any(fine.quantize(q, rounding=mode) == coarse
                   for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN))
    except InvalidOperation:
        return False


def number_equal(g, p, spec):
    ga, pa = numeric(g), numeric_prediction(p)
    if ga is None or pa is None:
        return False
    return units_equal(ga[1], pa[1], spec['question']) and rounded_equal(ga[0], pa[0], spec.get('decimals'))


def pages(s, ordered=False):
    return page_list(str(s).strip().strip('[]{}()'), ordered)


def table_page(s):
    t = surface(s)
    if re.search(r'\b(?:not|instead|either|or)\b', t):
        return None
    ids = re.findall(r'\btable\s+([a-z]?\d+[a-z]?|[a-z])(?!\w)', t)
    matches = re.findall(r'\bpages?\s*(\d+(?:\s*(?:-|to|,\s*(?:and\s+)?|and)\s*\d+)*)', t)
    if len(set(ids)) != 1 or not matches:
        return None
    ps = [pages(x) for x in matches]
    if any(x is None for x in ps) or any(x != ps[0] for x in ps):
        return None
    return (ids[0], ps[0])


def component_equal(g, p, spec):
    if numeric(g) is not None:
        return number_equal(g, p, spec)
    gi, pi = interval(g), interval(p)
    if gi:
        return pi is not None and all(number_equal(a, b, spec) for a, b in zip(gi, pi))
    return text_equal(g, p)


def components(spec):
    g = spec['gold']
    if spec.get('join_split'):  # e.g. author names with initials, whose commas are not list boundaries
        return g.split(spec['join_split'])
    return spec.get('components') or parts(g, spec.get('split_and', False))


def joined_components_equal(spec, pred):
    """All gold entities must be consumed; a gold substring inside a longer answer is not enough."""
    gs = tuple(lexical(x) for x in components(spec))
    target = lexical(pred)
    if not all(gs) or len(gs) > 16:
        return False
    ordered = spec['type'] == 'ordered_tuple'
    gnums, pnums = numeric_signature(spec['gold']), numeric_signature(pred)
    if (gnums != pnums if ordered else sorted(gnums) != sorted(pnums)):
        return False

    @lru_cache(None)
    def match(pos, remaining):
        if not remaining:
            return pos == len(target)
        choices = remaining[:1] if ordered else remaining
        for index in choices:
            if target.startswith(gs[index], pos):
                end = pos + len(gs[index])
                rest = tuple(i for i in remaining if i != index)
                if match(end, rest):
                    return True
                if rest and target.startswith('and', end) and match(end + 3, rest):
                    return True
        return False
    return match(0, tuple(range(len(gs))))


def taa_correct(spec, pred):
    """Type-aware correctness of an extracted answer. Returns (bool, reason)."""
    if strict_score(spec, pred):
        return True, 'strict_type_match'
    g, typ = spec['gold'], spec['type']
    for forms in EQUIVALENCES.get((spec['qid'], g), []):
        if any(text_key(pred) == text_key(a) for a in forms):
            return True, 'reviewed_equivalence'
    if typ == 'numeric' and number_equal(g, pred, spec):
        return True, 'rounding_or_numeric_format'
    if typ in ('page_set', 'page_tuple'):
        gg, pp = pages(g, typ == 'page_tuple'), pages(pred, typ == 'page_tuple')
        if pp is not None and gg == pp:
            return True, 'page_wrapper_format'
    if typ == 'table_page':
        gg, pp = table_page(g), table_page(pred)
        if gg is not None and gg == pp:
            return True, 'table_page_sentence_parsed'
    if typ in ('unordered_set', 'ordered_tuple'):
        if joined_components_equal(spec, pred):
            return True, 'components_equivalent_separators'
        if typ == 'ordered_tuple':
            gg = components(spec)
            for pp in [parts(pred), parts(pred, True), re.split(r'\s*[;|]\s*|\s+—\s+', str(pred))]:
                if len(gg) == len(pp) and all(component_equal(a, b, spec) for a, b in zip(gg, pp)):
                    return True, 'ordered_components_equivalent_values'
    if typ == 'interval':
        gi, pi = interval(g), interval(pred)
        if pi and all(number_equal(a, b, spec) for a, b in zip(gi, pi)):
            return True, 'interval_numeric_equivalence'
    if typ == 'text_exact' and text_equal(g, pred):
        return True, 'text_format_equivalence'
    return False, 'mismatch'


def answer_spec(row):
    spec = classify(row['qid'], row['question'], row['answer'])
    ov = SPEC_OVERRIDES.get(row['qid'])
    if ov and ov.get('gold') == row['answer']:
        spec.update({k: v for k, v in ov.items() if k != 'gold'})
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# 5. Optional gold-blind sentence post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _nz(s):
    return re.sub(r"[\s\"'“”‘’`*]+", " ", str(s).lower().replace("–", "-").replace("—", "-").replace("−", "-")).strip()


def grounded(extracted, response):
    """Accept a model-extracted short answer only if each part occurs verbatim in the response."""
    if not extracted or str(extracted).upper().startswith("NONE"):
        return False
    rr = " " + _nz(response) + " "
    return all(_nz(p) and _nz(p) in rr for p in re.split(r"\s*,\s+", str(extracted)))


def score_item(row, response, extracted=None):
    spec = answer_spec(row)
    ext = extract_answer(str(response))
    ok, reason = taa_correct(spec, ext)
    a = anls(row['answer'], ext)
    if extracted is not None and grounded(extracted, response):
        if not ok:
            ok2, reason2 = taa_correct(spec, str(extracted))
            if ok2:
                ok, reason = True, 'postprocessed_' + reason2
        a = max(a, anls(row['answer'], str(extracted)))
    return {'qid': row['qid'], 'answer_type': spec['type'], 'extracted_answer': ext,
            'taa': int(ok), 'taa_reason': reason, 'anls': a}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Command line
# ─────────────────────────────────────────────────────────────────────────────

PRED_KEYS = ('pred_answer', 'prediction', 'response', 'answer', 'pred')


def load_predictions(path, pred_key, extracted_key):
    text = open(path, encoding='utf-8').read()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'qid' not in obj:
            return {q: (v, None) for q, v in obj.items()}
    except json.JSONDecodeError:
        pass
    preds = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = pred_key or next((k for k in PRED_KEYS if k in r), None)
        if key is None or r.get(key) is None:
            continue
        preds[r['qid']] = (r[key], r.get(extracted_key) if extracted_key else None)
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True, help='data/test.jsonl of the release')
    ap.add_argument('--pred', required=True, help='predictions (JSON Lines with qid, or {qid: response})')
    ap.add_argument('--pred-key', help='field holding the raw response (auto-detected if omitted)')
    ap.add_argument('--extracted-key', help='optional field with a gold-blind extracted short answer')
    ap.add_argument('--out', help='write per-question scores (JSON Lines)')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding='utf-8') if l.strip()]
    preds = load_predictions(args.pred, args.pred_key, args.extracted_key)
    items, missing = [], 0
    for r in rows:
        if r['qid'] in preds:
            resp, ext = preds[r['qid']]
            it = score_item(r, resp, ext)
        else:
            missing += 1
            it = {'qid': r['qid'], 'answer_type': answer_spec(r)['type'], 'extracted_answer': None,
                  'taa': 0, 'taa_reason': 'missing_prediction', 'anls': 0.0}
        it.update(level=r.get('level'), category=r.get('category'))
        items.append(it)

    groups = defaultdict(list)
    for it in items:
        groups[('All', '')].append(it)
        groups[('Level', it['level'])].append(it)
        groups[('Category', it['category'])].append(it)
    print(f"questions {len(rows)}  predictions matched {len(rows) - missing}  missing {missing}")
    print(f"{'group':<48} {'n':>5} {'TAA':>7} {'ANLS':>7}")
    for kind in ('All', 'Level', 'Category'):
        for (k, name), its in sorted(groups.items()):
            if k != kind:
                continue
            taa = 100 * sum(i['taa'] for i in its) / len(its)
            an = 100 * sum(i['anls'] for i in its) / len(its)
            label = 'All' if kind == 'All' else f'{kind}: {name}'
            print(f"{label:<48} {len(its):>5} {taa:>7.2f} {an:>7.2f}")
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# 7. Reviewed data
# ─────────────────────────────────────────────────────────────────────────────

# Answer types reviewed by hand from the question and gold only (no predictions),
# for questions where the automatic type from `classify` is too coarse. Applied
# only while the gold answer is unchanged.
SPEC_OVERRIDES = {'PMC10157558_v5_3202724e': {'components': ['Ovid MEDLINE', 'Embase', 'APA PsycINFO', 'CINHAL', 'SCOPUS'],
                             'gold': 'Ovid MEDLINE, Embase, APA PsycINFO, CINHAL, SCOPUS',
                             'type': 'unordered_set'},
 'PMC10157558_v5_48c00e85': {'components': None,
                             'gold': '-10 and 11',
                             'split_and': None,
                             'type': 'text_exact'},
 'PMC10157558_v5_66f6f529': {'components': ['Ovid MEDLINE (.tw,kf)',
                                            'Embase (.tw,kw)',
                                            'APA PsycINFO (.ti,ab,id)'],
                             'gold': 'Ovid MEDLINE (.tw,kf), Embase (.tw,kw), APA PsycINFO (.ti,ab,id)',
                             'type': 'unordered_set'},
 'PMC10157558_v5_842b2199': {'gold': 'Search strategy; pages 4 and 5', 'type': 'section_page'},
 'PMC10157558_v5_8ed2fd5b': {'components': ['Speech Therapy', 'Occupational Therapy', 'Physical Therapy'],
                             'gold': 'Speech Therapy, Occupational Therapy, Physical Therapy',
                             'type': 'unordered_set'},
 'PMC10157558_v5_a245e6ff': {'gold': 'Search strategy; pages 4–5', 'type': 'section_page'},
 'PMC10157558_v5_a63f114e': {'components': ['Database', 'Search strategy'],
                             'gold': 'Database; Search strategy',
                             'type': 'unordered_set'},
 'PMC10157558_v5_dcca5b62': {'components': ['Text messag*', 'video conferenc*'],
                             'gold': 'Text messag* and video conferenc*',
                             'split_and': True,
                             'type': 'unordered_set'},
 'PMC10161420_v5_16c2d9df': {'components': ['Quite helpful', 'Very helpful'],
                             'gold': 'Quite helpful and Very helpful',
                             'split_and': True,
                             'type': 'unordered_set'},
 'PMC10164231_manual_83a7ac4d': {'components': ['Voluntary Service Providers', 'Management roles'],
                                 'gold': 'Voluntary Service Providers, Management roles',
                                 'type': 'ordered_tuple'},
 'PMC10164231_manual_b4b148d3': {'gold': '4, 8', 'split_and': None},
 'PMC10164231_v5_33bc7948': {'components': ['Managing changes...', '7'],
                             'gold': 'Managing changes..., 7',
                             'type': 'ordered_tuple'},
 'PMC10164231_v5_4c29e541': {'components': ['Clusters related to role of staff', 'Sample quotes'],
                             'gold': 'Clusters related to role of staff; Sample quotes',
                             'type': 'unordered_set'},
 'PMC10164231_v5_64df6745': {'components': ['Clusters…', 'Sample quotes'],
                             'gold': 'Clusters…; Sample quotes',
                             'type': 'unordered_set'},
 'PMC10169300_v5_8a6c97b1': {'components': ['4', 'Optimize workflows and access to information'],
                             'gold': '4; Optimize workflows and access to information',
                             'type': 'ordered_tuple'},
 'PMC10169884_v5_144d21ee': {'components': ['A', 'B', 'C'], 'gold': 'A, B, C', 'type': 'unordered_set'},
 'PMC10169884_v5_b9b6d59f': {'components': ['A', 'B', 'C', 'D'],
                             'gold': 'A, B, C, D',
                             'type': 'unordered_set'},
 'PMC10172242_v5_22ee7471': {'components': ['Short Form Health Survey 36 (SF-36)', '47'],
                             'gold': 'Short Form Health Survey 36 (SF-36), 47',
                             'type': 'ordered_tuple'},
 'PMC10172242_v5_2990ecde': {'components': ['European Quality of Life 5 Dimensions (EQ-5D)', '8'],
                             'gold': 'European Quality of Life 5 Dimensions (EQ-5D), 8',
                             'type': 'ordered_tuple'},
 'PMC10172242_v5_cf578595': {'components': ['SF-36 version 2', 'SF-36 subscale fatigue'],
                             'gold': 'SF-36 version 2; SF-36 subscale fatigue',
                             'type': 'unordered_set'},
 'PMC10178809_manual_23691387': {'gold': '114.65, 0.967', 'split_and': None},
 'PMC10178809_manual_24aea2c8': {'gold': '2.37, 0.660', 'split_and': None},
 'PMC10178809_manual_3a9dde48': {'gold': '49.58, 14.48', 'split_and': None},
 'PMC10178809_manual_3d5c709f': {'gold': '83, 0.776', 'split_and': None},
 'PMC10178809_manual_8fa49549': {'gold': '1.209, 0.655', 'split_and': None},
 'PMC10178809_manual_980ddcd5': {'gold': '4.19, 0.609', 'split_and': None},
 'PMC10178809_v5_491dcd12': {'components': ['Prejudice', 'Appreciation'],
                             'gold': 'Prejudice; Appreciation',
                             'type': 'unordered_set'},
 'PMC10178809_v5_620a6032': {'gold': '4.81, 0.578', 'split_and': None},
 'PMC10178809_v5_888efe97': {'components': ['Prejudice', 'Appreciation.'],
                             'gold': 'Prejudice and Appreciation.',
                             'split_and': True,
                             'type': 'unordered_set'},
 'PMC10178809_v5_8de84b93': {'gold': '4.61, 0.553', 'split_and': None},
 'PMC10178809_v5_9dce57ec': {'components': ['11N', '0.350'], 'gold': '11N, 0.350', 'type': 'ordered_tuple'},
 'PMC10178809_v5_a506ad4a': {'gold': '4.40, 0.537', 'split_and': None},
 'PMC10178809_v5_b2c05765': {'components': ['11N', '0.502.'], 'gold': '11N; 0.502.', 'type': 'ordered_tuple'},
 'PMC10178809_v5_b4416f6d': {'components': ['15P', '0.667'], 'gold': '15P, 0.667', 'type': 'ordered_tuple'},
 'PMC10179149_v5_562b9b7a': {'components': ['Bursell', 'S.-E. et al. [62] and Fonda', 'S.J. et al. [76]'],
                             'gold': 'Bursell, S.-E. et al. [62] and Fonda, S.J. et al. [76]',
                             'join_split': ' and ',
                             'split_and': True,
                             'type': 'unordered_set'},
 'PMC10179149_v5_f9dd9029': {'components': ['Li, R. et al. [ 67 ]', 'Li, Z. et al. [ 64 ]'],
                             'delimiter': ';',
                             'gold': 'Li, R. et al. [ 67 ]; Li, Z. et al. [ 64 ]',
                             'type': 'unordered_set'},
 'PMC10193889_v5_77088811': {'gold': 'Data analysis; 2', 'type': 'section_page'},
 'PMC10193889_v5_d609f787': {'gold': 'Study design; pages 129 and 130', 'type': 'section_page'},
 'PMC10193889_v5_fc509ec5': {'gold': '119 and 131', 'type': 'page_tuple'},
 'PMC10194850_v5_3f894b6f': {'gold': 'Descriptive findings; pages 4–5', 'type': 'section_page'},
 'PMC10194850_v5_74a61167': {'gold': 'Descriptive findings; 5', 'type': 'section_page'},
 'PMC10194850_v5_7f2227a8': {'components': ['Table 1', '2 pages'],
                             'gold': 'Table 1; 2 pages',
                             'type': 'ordered_tuple'},
 'PMC10194850_v5_d5b1c549': {'gold': 'Data collection; page 5', 'type': 'section_page'},
 'PMC10205023_manual_74c768df': {'components': ['R1', 'R4'], 'gold': 'R1, R4', 'type': 'unordered_set'},
 'PMC10213256_v5_c538b1a2': {'components': ['MYLK', 'MYL9'], 'gold': 'MYLK, MYL9', 'type': 'unordered_set'},
 'PMC10214026_v5_4e22f248': {'components': ['Sample Recruitment', 'Results'],
                             'gold': 'Sample Recruitment; Results',
                             'type': 'ordered_tuple'},
 'PMC10214026_v5_5b1125fc': {'components': ['Telemedicine', 'Other remote interventions'],
                             'gold': 'Telemedicine; Other remote interventions',
                             'type': 'unordered_set'},
 'PMC10216448_v5_32ff5073': {'components': ['Results', 'Table 1'],
                             'gold': 'Results; Table 1',
                             'type': 'ordered_tuple'},
 'PMC10216448_v5_4cc56d9c': {'components': ['Table 2', '5'], 'gold': 'Table 2, 5', 'type': 'ordered_tuple'},
 'PMC10221088_v5_286ec0c1': {'components': ['asparagine', '−15.913'],
                             'gold': 'asparagine, −15.913',
                             'type': 'ordered_tuple'},
 'PMC10221088_v5_dec3ec0d': {'components': ['salicylaldehyde', '2.909'],
                             'gold': 'salicylaldehyde, 2.909',
                             'type': 'ordered_tuple'},
 'PMC10235831_manual_259e0492': {'components': ['Health economics and service delivery', 'highest'],
                                 'gold': 'Health economics and service delivery, highest',
                                 'type': 'ordered_tuple'},
 'PMC10235831_manual_3e1accc9': {'components': ['Surgical treatment-forefoot',
                                                '"Review of long-term outcomes of implants (e.g., 1st '
                                                'metatarsophalangeal joint, interphlex, proximal '
                                                'interphalangeal joint)"'],
                                 'gold': 'Surgical treatment-forefoot, "Review of long-term outcomes of '
                                         'implants (e.g., 1st metatarsophalangeal joint, interphlex, '
                                         'proximal interphalangeal joint)"',
                                 'type': 'ordered_tuple'},
 'PMC10235831_manual_7e02a75b': {'gold': '6, 6', 'split_and': None},
 'PMC10235831_manual_d2b9ff2c': {'components': ['Rank', 'Agreement reached (%)'],
                                 'gold': 'Rank, Agreement reached (%)',
                                 'type': 'unordered_set'},
 'PMC10235831_manual_f2999922': {'components': ['Patient satisfaction and patient-reported outcomes', '4th'],
                                 'gold': 'Patient satisfaction and patient-reported outcomes, 4th',
                                 'type': 'ordered_tuple'},
 'PMC10235831_manual_fff4d101': {'components': ['Health economics and service delivery', '1st'],
                                 'gold': 'Health economics and service delivery, 1st',
                                 'type': 'ordered_tuple'},
 'PMC10307953_v5_9c5f654f': {'components': ['Fear of COVID-19', 'Risk perception'],
                             'gold': 'Fear of COVID-19; Risk perception',
                             'type': 'unordered_set'},
 'PMC10308762_manual_0e700b90': {'components': ['"A life-changing event"',
                                                '"Being alone"',
                                                '"Not speaking up"'],
                                 'gold': '"A life-changing event", "Being alone", "Not speaking up"',
                                 'type': 'unordered_set'},
 'PMC10308762_manual_16b79feb': {'components': ['DR7', 'DR9', 'DR10', 'DR6'],
                                 'gold': 'DR7, DR9, DR10, DR6',
                                 'type': 'unordered_set'},
 'PMC10308762_manual_85e0f1dd': {'components': ['"The tyranny of distance"',
                                                '"The systemic lack of resources in rural practice"',
                                                '"A fragmented health system"'],
                                 'gold': '"The tyranny of distance", "The systemic lack of resources in '
                                         'rural practice", "A fragmented health system"',
                                 'type': 'unordered_set'},
 'PMC10308762_manual_aa8953a7': {'components': ['DR6', 'DR4'], 'gold': 'DR6,DR4', 'type': 'unordered_set'},
 'PMC10308762_manual_cce9489a': {'components': ['Patient', 'Staff', 'System'],
                                 'gold': 'Patient, Staff, System',
                                 'type': 'unordered_set'},
 'PMC10308762_manual_d550c059': {'components': ['Patient', 'Staff', 'System'],
                                 'gold': 'Patient, Staff, System',
                                 'type': 'unordered_set'},
 'PMC10308762_manual_d7f380de': {'components': ['"Rural clinicians feel unsupported"',
                                                '"Enabling the training of rural clinicians in older trauma '
                                                'care"'],
                                 'gold': '"Rural clinicians feel unsupported", "Enabling the training of '
                                         'rural clinicians in older trauma care"',
                                 'type': 'ordered_tuple'},
 'PMC10308762_manual_fc3a7a06': {'components': ['DR9', 'DR10'], 'gold': 'DR9, DR10', 'type': 'unordered_set'},
 'PMC10318754_v5_aa2dd321': {'components': ['Superior', 'No difference', 'Inferior', 'Total'],
                             'gold': 'Superior, No difference, Inferior, Total',
                             'type': 'unordered_set'},
 'PMC10349219_manual_5f28d4ea': {'component_mode': 'text',
                                 'components': ['1.1-1.11', '2.1-2.10'],
                                 'gold': '1.1-1.11, 2.1-2.10',
                                 'type': 'ordered_tuple'},
 'PMC10349219_manual_70fc35f4': {'components': ['"Disconnected."',
                                                '"The perceived dispensability of recurrent miscarriage '
                                                'services and supports."'],
                                 'gold': '"Disconnected.", "The perceived dispensability of recurrent '
                                         'miscarriage services and supports."',
                                 'type': 'ordered_tuple'},
 'PMC10349219_manual_918af795': {'components': ['"Grieving in isolation."',
                                                '"The loss of supportive spaces"',
                                                '"Care in virtual spaces."'],
                                 'gold': '"Grieving in isolation.", "The loss of supportive spaces", "Care '
                                         'in virtual spaces."',
                                 'type': 'ordered_tuple'},
 'PMC10360290_v5_145fda22': {'components': ['Age (Year)', '61'],
                             'gold': 'Age (Year), 61',
                             'type': 'ordered_tuple'},
 'PMC10360290_v5_9264edbb': {'components': ['7', '<\u20090.0001'],
                             'gold': '7, <\u20090.0001',
                             'type': 'ordered_tuple'},
 'PMC10361730_v5_6fc06251': {'components': ['GO_CATION_CHANNEL_COMPLEX', '139'],
                             'gold': 'GO_CATION_CHANNEL_COMPLEX, 139',
                             'type': 'ordered_tuple'},
 'PMC10377367_manual_12de668c': {'components': ['Yamada et al.', 'Engelhardt et al.', 'Murray et al.'],
                                 'gold': 'Yamada et al., Engelhardt et al., Murray et al.',
                                 'type': 'ordered_tuple'},
 'PMC10377367_manual_14fb9fa7': {'components': ['Stavropoulos et al.', 'Murray et al.'],
                                 'gold': 'Stavropoulos et al., Murray et al.',
                                 'type': 'ordered_tuple'},
 'PMC10377367_manual_2e6a78aa': {'components': ['AQ-J',
                                                'Affinity for Hikikomori Scale',
                                                'Academic Failure Subscale',
                                                'Interpersonal Stress Event Scale',
                                                'Demographic interview',
                                                'PVGT'],
                                 'gold': 'AQ-J, Affinity for Hikikomori Scale, Academic Failure Subscale, '
                                         'Interpersonal Stress Event Scale, Demographic interview, PVGT',
                                 'type': 'ordered_tuple'},
 'PMC10377367_manual_386a56b1': {'components': ['Tateno [38]: 1038', 'Kondo: 337', 'Tateno [65]: 478'],
                                 'gold': 'Tateno [38]: 1038; Kondo: 337; Tateno [65]: 478',
                                 'type': 'ordered_tuple'},
 'PMC10377367_manual_60e39e0c': {'gold': '646, 420', 'split_and': None},
 'PMC10377367_manual_baf5da2f': {'gold': '22%, 9.1%', 'split_and': None},
 'PMC10392849_v5_c297f200': {'components': ['Relevant genotype/phenotype', 'Reference'],
                             'gold': 'Relevant genotype/phenotype; Reference',
                             'type': 'unordered_set'},
 'PMC10398984_manual_12a939e2': {'gold': '1.63E-05, 3.07E-07', 'split_and': None},
 'PMC10398984_manual_4b1586ab': {'components': ['"Response to hypoxia."',
                                                '"Positive regulation of apoptotic process."'],
                                 'gold': '"Response to hypoxia.",  "Positive regulation of apoptotic '
                                         'process."',
                                 'type': 'ordered_tuple'},
 'PMC10398984_manual_72ac618c': {'components': ['"Positive regulation of angiogenesis."', '"CD36 molecule."'],
                                 'gold': '"Positive regulation of angiogenesis.", "CD36 molecule."',
                                 'type': 'ordered_tuple'},
 'PMC10398984_manual_85aac5ce': {'gold': '5, 2', 'split_and': None},
 'PMC10398984_manual_8ab21766': {'components': ['"Positive regulation of ERK1 and ERK2 cascade"',
                                                '"Inflammatory response"',
                                                'Table 4 ("Inflammatory response")'],
                                 'gold': '"Positive regulation of ERK1 and ERK2 cascade", "Inflammatory '
                                         'response", Table 4 ("Inflammatory response")',
                                 'type': 'ordered_tuple'},
 'PMC10398984_manual_9d59b412': {'components': ['"Angiogenesis."',
                                                '"Positive regulation of inflammatory response."'],
                                 'gold': '"Angiogenesis.", "Positive regulation of inflammatory response."',
                                 'type': 'ordered_tuple'},
 'PMC10398984_manual_ae02a0bf': {'components': ['"Positive regulation of I-kappaB kinase/NF-kappaB '
                                                'signaling."',
                                                '"Inflammatory response."'],
                                 'gold': '"Positive regulation of I-kappaB kinase/NF-kappaB signaling.", '
                                         '"Inflammatory response."',
                                 'type': 'ordered_tuple'},
 'PMC10398984_manual_ddc19c16': {'components': ['"Integrin-mediated signaling pathway."',
                                                '"Leukocyte cell–cell adhesion."'],
                                 'gold': '"Integrin-mediated signaling pathway.", "Leukocyte cell–cell '
                                         'adhesion."',
                                 'type': 'ordered_tuple'},
 'PMC10398984_manual_e1fb019d': {'components': ['"Leukocyte cell–cell adhesion."',
                                                '"Vascular cell adhesion molecule 1."'],
                                 'gold': '"Leukocyte cell–cell adhesion.", "Vascular cell adhesion molecule '
                                         '1."',
                                 'type': 'ordered_tuple'},
 'PMC10398984_manual_e4a72329': {'components': ['"Positive regulation of angiogenesis."', '"Aging."'],
                                 'gold': '"Positive regulation of angiogenesis.", "Aging."',
                                 'type': 'ordered_tuple'},
 'PMC10398984_manual_e89addf8': {'components': ['"Positive regulation of ERK1 and ERK2 cascade."',
                                                '"Positive regulation of cell proliferation,"',
                                                '"Positive regulation of apoptotic process."',
                                                '"Myeloid dendritic cell differentiation."'],
                                 'gold': '"Positive regulation of ERK1 and ERK2 cascade.", "Positive '
                                         'regulation of cell proliferation,", "Positive regulation of '
                                         'apoptotic process.", "Myeloid dendritic cell differentiation."',
                                 'type': 'ordered_tuple'},
 'PMC10420670_manual_198f3960': {'components': ['Soft Drinks', 'Fruit Drinks', 'Sport and Energy Drinks'],
                                 'gold': 'Soft Drinks, Fruit Drinks, Sport and Energy Drinks',
                                 'type': 'unordered_set'},
 'PMC10420670_manual_219a0b63': {'gold': '2.08, 4.83', 'split_and': None},
 'PMC10420670_manual_e88eb80f': {'gold': '2.26, 63.98', 'split_and': None},
 'PMC10454118_manual_95e09d3d': {'gold': '8109, 42.66', 'split_and': None},
 'PMC10454118_manual_e882a2da': {'gold': '40.5%, 185', 'split_and': None},
 'PMC10474523_manual_1f33080a': {'components': ['Vitamin D deficiency', '117'],
                                 'gold': 'Vitamin D deficiency, 117',
                                 'type': 'ordered_tuple'},
 'PMC10474523_manual_33a8fda8': {'components': ['Third trimester', '56'],
                                 'gold': 'Third trimester, 56',
                                 'type': 'ordered_tuple'},
 'PMC10474523_manual_a025bae6': {'gold': '57, 2.16', 'split_and': None},
 'PMC10474523_manual_adbb4220': {'components': ['2-5 times', '70'],
                                 'gold': '2-5 times, 70',
                                 'type': 'ordered_tuple'},
 'PMC10474523_manual_ff52acc4': {'gold': 'Five criteria, 122', 'split_and': None},
 'PMC10556542_manual_28a596e8': {'gold': '0.018, 0.663', 'split_and': None},
 'PMC10556542_manual_756d7baa': {'gold': '23.00%, 103 KU/L', 'split_and': None},
 'PMC10556542_manual_c4ea0d8f': {'gold': '13, 13.40%, 411 KU/L', 'split_and': None},
 'PMC10572221_manual_59d0f9d2': {'gold': '7.694, 8.206', 'split_and': None},
 'PMC10572221_manual_c5ee4a6e': {'gold': '1.822, 1.806', 'split_and': None},
 'PMC10572221_manual_c64d2786': {'gold': '3.071, 4.031', 'split_and': None},
 'PMC10661818_manual_128d076d': {'components': ['Midbrain', '11.0%'],
                                 'gold': 'Midbrain, 11.0%',
                                 'type': 'ordered_tuple'},
 'PMC10661818_manual_309f65ce': {'components': ['Brain atrophy', '16'],
                                 'gold': 'Brain atrophy, 16',
                                 'type': 'ordered_tuple'},
 'PMC10661818_manual_ce94ec00': {'components': ['11.0%', 'Lobar'],
                                 'gold': '11.0%, Lobar',
                                 'type': 'ordered_tuple'},
 'PMC10702707_manual_3b2357ee': {'gold': '3, 4', 'type': 'page_tuple'},
 'PMC10702707_manual_bcd7f622': {'gold': 'page2, page4', 'type': 'page_tuple'},
 'PMC10768938_manual_0d9ce4fa': {'components': ['Variables', 'Frequency (n=91)', 'Percentage (%)'],
                                 'gold': 'Variables, Frequency (n=91), Percentage (%)',
                                 'type': 'unordered_set'},
 'PMC10768938_manual_4fddc267': {'gold': '0.002, 83.5%, 16.5%', 'split_and': None},
 'PMC10768938_manual_548781bb': {'components': ['BMI 25–29.9', '16.1%'],
                                 'gold': 'BMI 25–29.9, 16.1%',
                                 'type': 'ordered_tuple'},
 'PMC10905661_manual_8edeb8fe': {'components': ['1 Oct–25 Nov 2021', '1 Nov–26 Dec 2021'],
                                 'gold': '1 Oct–25 Nov 2021, 1 Nov–26 Dec 2021',
                                 'type': 'ordered_tuple'},
 'PMC10905661_manual_8ff87d1a': {'components': ['71%', 'NA'], 'gold': '71%, NA', 'type': 'ordered_tuple'},
 'PMC10905661_manual_f919b18d': {'components': ['71%', 'NA'], 'gold': '71%, NA', 'type': 'ordered_tuple'},
 'PMC11083360__table1__condition_lookup': {'components': ['GP6871LB', 'GP6872LB'],
                                           'gold': 'GP6871LB\nGP6872LB',
                                           'type': 'unordered_set'},
 'PMC11088816_hd_d956a9d5': {'components': ['A. baumannii', 'E. faecalis'],
                             'gold': 'A. baumannii, E. faecalis',
                             'type': 'unordered_set'},
 'PMC11119127_manual_328f1e1d': {'gold': 'Table 1, Table 2, Table 3, Table 4', 'type': 'table_tuple'},
 'PMC11119127_manual_6028ecfc': {'components': ['Histological Type (Squamous)',
                                                'Clinical Presentation (cN0, cN+)',
                                                'Figo Stage (I–IVB)',
                                                'Focality (Unifocal, Multifocal)',
                                                'Size of Primary Lesion',
                                                'Grading (G1–G3)',
                                                'Treatment',
                                                'SUV Max and Range'],
                                 'gold': 'Histological Type (Squamous), Clinical Presentation (cN0, cN+), '
                                         'Figo Stage (I–IVB), Focality (Unifocal, Multifocal), Size of '
                                         'Primary Lesion, Grading (G1–G3), Treatment, SUV Max and Range',
                                 'type': 'unordered_set'},
 'PMC11435373_manual_4b5cf449': {'components': ['Cistus parviflorus', 'C. monspeliasis.'],
                                 'gold': 'Cistus parviflorus, C. monspeliasis.',
                                 'type': 'ordered_tuple'},
 'PMC11435373_manual_5bc40e5c': {'components': ['Leaves', 'Twigs', 'Roots'],
                                 'gold': 'Leaves, Twigs, Roots',
                                 'type': 'unordered_set'},
 'PMC11435373_manual_70a11f31': {'components': ['Punicalagin', 'Punicalagin isomer'],
                                 'gold': 'Punicalagin, Punicalagin isomer',
                                 'type': 'unordered_set'},
 'PMC11435373_manual_e8ef8cd6': {'gold': '5, 6', 'split_and': None},
 'PMC11507563_hd_9a6c6dc7': {'components': ['Sweet', 'nutty'],
                             'gold': 'Sweet, nutty',
                             'type': 'unordered_set'},
 'PMC11565214_manual_aef0b80c': {'gold': '47, 26.6, 147, 46.2', 'split_and': None},
 'PMC11565214_manual_d09d2171': {'gold': '299290, 299290', 'split_and': None},
 'PMC11759003_manual_5da015f7': {'components': ['Main model',
                                                'With autocorrelation',
                                                'With relative humidity'],
                                 'gold': 'Main model, With autocorrelation, With relative humidity',
                                 'type': 'unordered_set'},
 'PMC11759003_manual_8ae75a7d': {'components': ['Toyama', '46.5%'],
                                 'gold': 'Toyama, 46.5%',
                                 'type': 'ordered_tuple'},
 'PMC11759003_manual_ba140d85': {'components': ['Okinawa', '55'],
                                 'gold': 'Okinawa, 55',
                                 'type': 'ordered_tuple'},
 'PMC11942188_manual_92e641b6': {'components': ['Land recovery (Ñukemapu, DM1)',
                                                'dump closure (DM2)',
                                                'water conservation (HM3)',
                                                'removing eucalyptus/pine (GF1)',
                                                'chemical-free farming (GF1)',
                                                '40h work week (GF2)',
                                                'knowledge exchange (DM1).'],
                                 'gold': 'Land recovery (Ñukemapu, DM1), dump closure (DM2), water '
                                         'conservation (HM3), removing eucalyptus/pine (GF1), chemical-free '
                                         'farming (GF1), 40h work week (GF2), knowledge exchange (DM1).',
                                 'type': 'unordered_set'}}

# Reviewed alternative surface forms of the same answer, keyed by (qid, gold).
# Each was accepted after reviewing question and gold, and applies to every model.
EQUIVALENCES = {('PMC10157558_v5_107638a6', 'A systematic review and meta-analysis'): [['A systematic review (and '
                                                                         'meta-analysis)'],
                                                                        ['Systematic review and '
                                                                         'meta-analysis']],
 ('PMC10157558_v5_3202724e', 'Ovid MEDLINE, Embase, APA PsycINFO, CINHAL, SCOPUS'): [['Ovid MEDLINE, Embase, '
                                                                                      'APA PsycINFO, CINAHL, '
                                                                                      'SCOPUS'],
                                                                                     ['Ovid MEDLINE, Embase, '
                                                                                      'APA PsycINFO, CINAHL, '
                                                                                      'and Scopus.'],
                                                                                     ['Ovid MEDLINE, Embase, '
                                                                                      'APA PsycINFO, CINAHL, '
                                                                                      'and SCOPUS.'],
                                                                                     ['Ovid MEDLINE, Embase, '
                                                                                      'APA PsychINFO, '
                                                                                      'CINAHL, and Scopus']],
 ('PMC10157558_v5_5a3a1abf', '-(ehealth or e-health).ti,ab,id'): [['(ehealth or e-health).ti,ab,id.'],
                                                                  ['(e-health or ehealth).ti,ab,id'],
                                                                  ['(ehealth or e-health).ti,ab,id']],
 ('PMC10157558_v5_e4858c21', '((online or web or remote* or virtual or digital) adj1 (intervention* or therap* or aftercare or rehab* or consult*))'): [['(online '
                                                                                                                                                         'or '
                                                                                                                                                         'web '
                                                                                                                                                         'or '
                                                                                                                                                         'remote* '
                                                                                                                                                         'or '
                                                                                                                                                         'virtual '
                                                                                                                                                         'or '
                                                                                                                                                         'digital) '
                                                                                                                                                         'adj1 '
                                                                                                                                                         '(intervention* '
                                                                                                                                                         'or '
                                                                                                                                                         'therap* '
                                                                                                                                                         'or '
                                                                                                                                                         'aftercare '
                                                                                                                                                         'or '
                                                                                                                                                         'rehab* '
                                                                                                                                                         'or '
                                                                                                                                                         'consult*)']],
 ('PMC10157558_v5_f1b9337f', '8. occupational therapist/or physical therapist/or speech therapist/'): [['8. '
                                                                                                        'occupational '
                                                                                                        'therapist/ '
                                                                                                        'or '
                                                                                                        'physical '
                                                                                                        'therapist/ '
                                                                                                        'or '
                                                                                                        'speech '
                                                                                                        'therapist/']],
 ('PMC10164231_v5_3e5d2cd2', 'Managing changes to new locally based models'): [['Managing changes to new '
                                                                                'locally based service '
                                                                                'models']],
 ('PMC10164231_v5_4c29e541', 'Clusters related to role of staff; Sample quotes'): [['Clusters related to '
                                                                                    'role of staff and '
                                                                                    'Sample quotes.'],
                                                                                   ['Clusters related to '
                                                                                    'role of staff | Sample '
                                                                                    'quotes']],
 ('PMC10164231_v5_696e1da1', 'All location types — 16'): [['All location types, 16.']],
 ('PMC10164231_v5_6c8eba13', 'Explanatory sequential mixed methods'): [['explanatory sequential mixed '
                                                                        'methods design'],
                                                                       ['Explanatory sequential mixed '
                                                                        'methods design.'],
                                                                       ['Explanatory sequential mixed '
                                                                        'methods design']],
 ('PMC10169300_v5_0c075541', 'Table 1 by 1'): [['Table 1, by 1']],
 ('PMC10169884_v5_4cfe8307', 'Table 1 by 2 pages'): [['Table 1, by 2 pages.']],
 ('PMC10172242_v5_0ec59398', 'EQ-5D-5 Levels'): [['EQ-5D-5L Levels']],
 ('PMC10178809_v5_776cd862', 'Appreciation on 15P'): [['Appreciation, 15P'], ['Appreciation, item 15P']],
 ('PMC10179149_v5_562b9b7a', 'Bursell, S.-E. et al. [62] and Fonda, S.J. et al. [76]'): [['Bursell, S-E. et '
                                                                                          'al. [62] and '
                                                                                          'Fonda, S.J. et '
                                                                                          'al. [76]']],
 ('PMC10179149_v5_f9dd9029', 'Li, R. et al. [ 67 ]; Li, Z. et al. [ 64 ]'): [['Li, R. et al. [67] and Li, Z. '
                                                                              'et al. [64]']],
 ('PMC10193889_v5_3b5a156e', 'The organisation of primary care'): [['The organization of primary care']],
 ('PMC10193889_v5_a6cee22b', 'Creating a positive climate of change'): [['Create a positive climate of '
                                                                         'change'],
                                                                        ['F: Creating a positive climate of '
                                                                         'change']],
 ('PMC10193889_v5_bf9b5ac0', 'Nurses'): [['Nurse']],
 ('PMC10193889_v5_c427acea', 'FGPT2 with 6 participants'): [['FGPT2, 6 participants']],
 ('PMC10193889_v5_dd594cc8', 'Feeling guilt and shame'): [['B: Feeling guilt and shame']],
 ('PMC10194850_v5_321dbc0e', '2'): [["Two sites list 'Balloon catheter' in 'Methods for home CR'."]],
 ('PMC10194850_v5_c3289d27', '5'): [['Five sites list midwives among those who do CR.']],
 ('PMC10214026_v5_5b1125fc', 'Telemedicine; Other remote interventions'): [['Telemedicine Other remote '
                                                                            'interventions']],
 ('PMC10216448_v5_12c3c45d', 'c.5266dupC (6)'): [['c.5266dupC, 6']],
 ('PMC10216448_v5_4a176e40', 'c.9371A>T (3)'): [['c.9371A>T, 3']],
 ('PMC10218104_v5_6346ce9c', '1'): [['1 Nora, an undergraduate student taking the course for three credits, '
                                     'started the course focusing on personal actions. Nora reflected on the '
                                     'impact of climate change but was vague about how to counter the '
                                     'impact. In their module two reflection, Nora identified specific '
                                     'actions they can take as a health professional, in places, sharing '
                                     'care-related actions, but also considering how they could communicate '
                                     'with their patients regarding specific, health-related actions they '
                                     'can take. Through the course, Nora showed a growing awareness of '
                                     'professional actions. In addition to the of the interest development '
                                     'framework, as they engaged with and reflected deepening knowledge of '
                                     'the course content.']],
 ('PMC10235831_manual_f2999922', 'Patient satisfaction and patient-reported outcomes, 4th'): [['Patient '
                                                                                               'satisfaction '
                                                                                               'and patient '
                                                                                               'reported '
                                                                                               'outcome '
                                                                                               'measures; '
                                                                                               '4th']],
 ('PMC10277563_v5_973b406d', '4'): [['Four CFIR domains.']],
 ('PMC10289277_v5_b45f2155', '2'): [['Two studies used Swept-source OCT (SS-OCT).']],
 ('PMC10307953_v5_9c5f654f', 'Fear of COVID-19; Risk perception'): [['Fear of COVID-19 and risk '
                                                                     'perception.']],
 ('PMC10308762_manual_acaa2c96', '"Multidisciplinary and coordinated care as standard of care", "A coordinator for inpatient and post-acute care", "Enabling the training of rural clinicians in older trauma care"'): [['Multidisciplinary '
                                                                                                                                                                                                                         'and '
                                                                                                                                                                                                                         'coordinated '
                                                                                                                                                                                                                         'care '
                                                                                                                                                                                                                         'as '
                                                                                                                                                                                                                         'standard '
                                                                                                                                                                                                                         'of '
                                                                                                                                                                                                                         'care, '
                                                                                                                                                                                                                         'A '
                                                                                                                                                                                                                         'coordinator '
                                                                                                                                                                                                                         'for '
                                                                                                                                                                                                                         'inpatient '
                                                                                                                                                                                                                         'and '
                                                                                                                                                                                                                         'post-acute '
                                                                                                                                                                                                                         'care, '
                                                                                                                                                                                                                         'Enabling '
                                                                                                                                                                                                                         'the '
                                                                                                                                                                                                                         'training '
                                                                                                                                                                                                                         'of '
                                                                                                                                                                                                                         'rural '
                                                                                                                                                                                                                         'clinicians '
                                                                                                                                                                                                                         'in '
                                                                                                                                                                                                                         'older '
                                                                                                                                                                                                                         'trauma '
                                                                                                                                                                                                                         'care.'],
                                                                                                                                                                                                                        ['Multidisciplinary '
                                                                                                                                                                                                                         'and '
                                                                                                                                                                                                                         'coordinated '
                                                                                                                                                                                                                         'care '
                                                                                                                                                                                                                         'as '
                                                                                                                                                                                                                         'standard '
                                                                                                                                                                                                                         'of '
                                                                                                                                                                                                                         'care, '
                                                                                                                                                                                                                         'A '
                                                                                                                                                                                                                         'coordinator '
                                                                                                                                                                                                                         'for '
                                                                                                                                                                                                                         'in-patient '
                                                                                                                                                                                                                         'and '
                                                                                                                                                                                                                         'post-acute '
                                                                                                                                                                                                                         'care, '
                                                                                                                                                                                                                         'Enabling '
                                                                                                                                                                                                                         'the '
                                                                                                                                                                                                                         'training '
                                                                                                                                                                                                                         'of '
                                                                                                                                                                                                                         'rural '
                                                                                                                                                                                                                         'clinicians '
                                                                                                                                                                                                                         'in '
                                                                                                                                                                                                                         'older '
                                                                                                                                                                                                                         'trauma '
                                                                                                                                                                                                                         'care']],
 ('PMC10308762_manual_d7f380de', '"Rural clinicians feel unsupported", "Enabling the training of rural clinicians in older trauma care"'): [['Rural '
                                                                                                                                             'clinicians '
                                                                                                                                             'feel '
                                                                                                                                             'unsupported '
                                                                                                                                             'and '
                                                                                                                                             'Enabling '
                                                                                                                                             'the '
                                                                                                                                             'training '
                                                                                                                                             'of '
                                                                                                                                             'rural '
                                                                                                                                             'clinicians '
                                                                                                                                             'in '
                                                                                                                                             'older '
                                                                                                                                             'trauma '
                                                                                                                                             'care.'],
                                                                                                                                            ['Rural '
                                                                                                                                             'clinicians '
                                                                                                                                             'feel '
                                                                                                                                             'unsupported '
                                                                                                                                             'and '
                                                                                                                                             'Enabling '
                                                                                                                                             'the '
                                                                                                                                             'training '
                                                                                                                                             'of '
                                                                                                                                             'rural '
                                                                                                                                             'clinicians '
                                                                                                                                             'in '
                                                                                                                                             'older '
                                                                                                                                             'trauma '
                                                                                                                                             'care']],
 ('PMC10333690_v5_61238787', 'TABLE 1 Continued'): [['TABLE 1 | Continued']],
 ('PMC10341104_v5_cd1d3de0', '5 (42)'): [['5 (42%)']],
 ('PMC10386879__table1__dominant_col4', 'Platelets (130–400 × 10 3 /µL)'): [['Platelets (130-400 x 10³/µL)'],
                                                                            ['Platelets (130-400 x 10³/μL)']],
 ('PMC10392849_v5_c297f200', 'Relevant genotype/phenotype; Reference'): [['Relevant '
                                                                          'genotype/phenotypeReference']],
 ('PMC10398984_manual_56837c29', 'Lipopolysaccharide-mediated singaling pathway'): [['Lipopolysaccharide-mediated '
                                                                                     'signaling pathway']],
 ('PMC10474523_manual_67167fed', 'Suffering from insomnia, 205'): [['Suffering from insomnia; 205']],
 ('PMC10474523_manual_ff52acc4', 'Five criteria, 122'): [['Five criteria*, 122']],
 ('PMC10513325_hd_b5ae3e52', 'PTA Ear: both'): [['PTA (Ear: both)']],
 ('PMC10513325_manual_738be54e', 'Ogorodnikova et al. (2017)'): [['Ogorodnikova et al. (2017) [37]']],
 ('PMC10556542_manual_890b3eab', 'Group2'): [['Group 2.']],
 ('PMC10572221_manual_03dd267e', 'OS: overall survival.'): [['overall survival'], ['Overall survival']],
 ('PMC10572221_manual_cb4414f8', 'Multivariate OS — "Model before Stepwise Selection" and "Model after Stepwise Selection."'): [['Multivariate '
                                                                                                                                 'OS, '
                                                                                                                                 'Model '
                                                                                                                                 'before '
                                                                                                                                 'Stepwise '
                                                                                                                                 'Selection, '
                                                                                                                                 'Model '
                                                                                                                                 'after '
                                                                                                                                 'Stepwise '
                                                                                                                                 'Selection'],
                                                                                                                                ['Multivariate '
                                                                                                                                 'OS, '
                                                                                                                                 'Model '
                                                                                                                                 'before '
                                                                                                                                 'Stepwise '
                                                                                                                                 'Selection, '
                                                                                                                                 'and '
                                                                                                                                 'Model '
                                                                                                                                 'after '
                                                                                                                                 'Stepwise '
                                                                                                                                 'Selection.'],
                                                                                                                                ['Multivariate, '
                                                                                                                                 'Model '
                                                                                                                                 'before '
                                                                                                                                 'Stepwise '
                                                                                                                                 'Selection, '
                                                                                                                                 'and '
                                                                                                                                 'Model '
                                                                                                                                 'after '
                                                                                                                                 'Stepwise '
                                                                                                                                 'Selection.']],
 ('PMC10573664_hd_21c171b7', 'Polymerization with “LaboLight DUO” curing unit for 20 min.'): [['GC TEMP '
                                                                                               'PRINT; '
                                                                                               'Polymerization '
                                                                                               'with '
                                                                                               '“LaboLight '
                                                                                               'DUO” curing '
                                                                                               'unit for 20 '
                                                                                               'min.']],
 ('PMC10661818_manual_309f65ce', 'Brain atrophy, 16'): [['Brain atrophy - 16']],
 ('PMC10690111__table2__condition_lookup', '30, Female / 24, Male'): [['30, Female; 24, Male'],
                                                                      ['30, Female and 24, Male'],
                                                                      ['30, Female\n24, Male']],
 ('PMC10905661_manual_e2f1b008', '3'): [['Three distinct booster levels are reported in each table.'],
                                        ['Three distinct booster levels are reported in each table: first '
                                         'booster, second booster, and third booster.']],
 ('PMC11015693_xp_6e0d5c40', '6.3. Equipment and Resources'): [['Equipment and resources']],
 ('PMC11273413_xp_0d7db15e', 'Generalized Anxiety Disorder 7 (GAD—7 [ 31 ])'): [['The Generalized Anxiety '
                                                                                 'Disorder 7 (GAD—7 [ 31 ] '
                                                                                 ')'],
                                                                                ['The Generalized Anxiety '
                                                                                 'Disorder 7 (GAD-7 [ 31 '
                                                                                 '])']],
 ('PMC11435373_manual_453670ad', 'Prodelphinidin B isomers'): [['Prodelphinidin B isomer 1'],
                                                               ['Prodelphinidin B isomer']],
 ('PMC11581225_hd_766d03da', 'Unknowns'): [['Unknown'], ['unknown']],
 ('PMC11694991_xp_83d41a31', 'Suarez et al , 2020'): [['Suarez, et al 2020']],
 ('PMC11759003_manual_3e032ba7', 'Saitama,15.8°C'): [['Saitama; 15.8°C'], ['Saitama, 15.8 °C']],
 ('PMC11759003_manual_ceeebe70', 'Okinawa,13.9%'): [['Okinawa, 13.9']],
 ('PMC12104190_xp_d76a6ed0', "Training is a special pedagogical and sporting process.'"): [['Training is a '
                                                                                            'special '
                                                                                            'pedagogical and '
                                                                                            'sporting '
                                                                                            'process'],
                                                                                           ['Training is a '
                                                                                            'special '
                                                                                            'pedagogical and '
                                                                                            'sporting '
                                                                                            'process.']],
 ('PMC12350084_xp_d8ebe0a5', '4. Clinical teams'): [['Clinical teams']]}


if __name__ == '__main__':
    sys.exit(main())
