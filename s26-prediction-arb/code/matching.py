import re
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# optional heavy deps -- don't crash if they're missing
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    from rapidfuzz import fuzz
    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False


@dataclass
class MatchResult:
    is_match: bool
    confidence: float
    method: str          # 'rule', 'embedding', 'fuzzy', 'llm', 'rejected'
    latency_ms: float


# ---- field extraction ----

# kalshi tickers have a pretty rigid format
# sports: KXNFLGAME-24DEC07-CHIDAL
# crypto: KXBTC-24DEC07-T50000
# general: KX<category>-<date>-<params>

_SPORTS_PAT = re.compile(r'^KX([A-Z]+)GAME-(\d{2})([A-Z]{3})(\d{2})-([A-Z]+)$')
_CRYPTO_PAT = re.compile(r'^KX([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-T(\d+)$')
_GENERAL_PAT = re.compile(r'^KX([A-Z]+)-(\d{2})([A-Z]{3})-?(\d{0,2})-?(.*)$')

_MONTH_MAP = {
    'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
    'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
    'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12',
}

_MONTH_NAMES = {
    'JAN': 'january', 'FEB': 'february', 'MAR': 'march', 'APR': 'april',
    'MAY': 'may', 'JUN': 'june', 'JUL': 'july', 'AUG': 'august',
    'SEP': 'september', 'OCT': 'october', 'NOV': 'november', 'DEC': 'december',
}


def _parse_date(yy, mon_str, dd):
    mm = _MONTH_MAP.get(mon_str, '??')
    return f"20{yy}-{mm}-{dd}"


def extract_kalshi_fields(ticker: str) -> Optional[Dict]:
    # try sports first
    m = _SPORTS_PAT.match(ticker)
    if m:
        league, yy, mon, dd, teams_str = m.groups()
        teams = [teams_str[i:i+3] for i in range(0, len(teams_str), 3)]
        return {
            'category': league.lower(),
            'teams': teams,
            'date_str': _parse_date(yy, mon, dd),
            'threshold': None,
            'raw_ticker': ticker,
        }

    # crypto
    m = _CRYPTO_PAT.match(ticker)
    if m:
        symbol, yy, mon, dd, thresh = m.groups()
        return {
            'category': 'crypto',
            'symbol': symbol,
            'teams': [],
            'date_str': _parse_date(yy, mon, dd),
            'threshold': float(thresh),
            'raw_ticker': ticker,
        }

    # general fallback -- catches politics, econ, etc
    m = _GENERAL_PAT.match(ticker)
    if m:
        cat, yy, mon, dd, params = m.groups()
        threshold = None
        if params and params.startswith('T'):
            try:
                threshold = float(params[1:])
            except ValueError:
                pass
        return {
            'category': cat.lower(),
            'teams': [],
            'date_str': _parse_date(yy, mon, dd if dd else '01'),
            'threshold': threshold,
            'raw_ticker': ticker,
        }

    return None


# polymarket descriptions are free text, so we have to be a bit creative

_CATEGORY_KEYWORDS = {
    'nfl': 'nfl', 'nba': 'nba', 'mlb': 'mlb', 'nhl': 'nhl',
    'bitcoin': 'crypto', 'btc': 'crypto', 'ethereum': 'crypto', 'eth': 'crypto',
    'solana': 'crypto', 'sol': 'crypto',
    'president': 'politics', 'election': 'politics', 'vote': 'politics',
    'governor': 'politics', 'senate': 'politics',
    'fed rate': 'economics', 'interest rate': 'economics', 'fomc': 'economics',
    'gdp': 'economics', 'unemployment': 'economics',
    'chiefs': 'nfl', 'bills': 'nfl', '49ers': 'nfl', 'eagles': 'nfl',
    'cowboys': 'nfl', 'ravens': 'nfl', 'lions': 'nfl', 'dolphins': 'nfl',
    'celtics': 'nba', 'nuggets': 'nba', 'bucks': 'nba', 'suns': 'nba',
    'warriors': 'nba', 'lakers': 'nba', 'timberwolves': 'nba', 'thunder': 'nba',
    'dodgers': 'mlb', 'braves': 'mlb', 'astros': 'mlb', 'yankees': 'mlb',
    'rangers': 'mlb', 'orioles': 'mlb',
}

_VERSUS_PAT = re.compile(r'(\w[\w\s]*?)\s+(?:beat|vs\.?|versus)\s+(\w[\w\s]*?)(?:\s|$|[,?.])', re.IGNORECASE)
_DATE_PAT = re.compile(r'(\w+)\s+(\d{1,2}),?\s*(\d{4})')
_THRESHOLD_PAT = re.compile(r'(?:above|over|>)\s*\$?([\d,]+(?:\.\d+)?)', re.IGNORECASE)


def extract_polymarket_fields(description: str) -> Dict:
    desc_lower = description.lower()

    # category
    category = None
    for kw, cat in _CATEGORY_KEYWORDS.items():
        if kw in desc_lower:
            category = cat
            break

    # teams / entities
    teams = []
    vm = _VERSUS_PAT.search(description)
    if vm:
        teams = [vm.group(1).strip(), vm.group(2).strip()]

    # date
    date_str = None
    dm = _DATE_PAT.search(description)
    if dm:
        month_name, day, year = dm.groups()
        month_abbr = month_name[:3].upper()
        mm = _MONTH_MAP.get(month_abbr, '??')
        date_str = f"{year}-{mm}-{day.zfill(2)}"

    # threshold
    threshold = None
    tm = _THRESHOLD_PAT.search(description)
    if tm:
        threshold = float(tm.group(1).replace(',', ''))

    return {
        'category': category,
        'teams': teams,
        'date_str': date_str,
        'threshold': threshold,
    }


# ---- matching primitives ----

def rule_based_match(k_fields: Dict, p_fields: Dict) -> float:
    # weighted score from structured field comparison
    score = 0.0

    # category (0.3)
    if k_fields.get('category') and p_fields.get('category'):
        # map sports subcategories
        k_cat = k_fields['category']
        p_cat = p_fields['category']
        if k_cat == p_cat:
            score += 0.3
        elif {k_cat, p_cat} <= {'nfl', 'nba', 'mlb', 'nhl'}:
            pass  # different sports, no credit

    # teams/entities (0.3)
    k_teams = set(t.lower() for t in (k_fields.get('teams') or []))
    p_teams = set(t.lower() for t in (p_fields.get('teams') or []))
    if k_teams and p_teams:
        overlap = len(k_teams & p_teams)
        total = max(len(k_teams | p_teams), 1)
        score += 0.3 * (overlap / total)

    # date (0.2)
    if k_fields.get('date_str') and p_fields.get('date_str'):
        try:
            from datetime import datetime
            kd = datetime.strptime(k_fields['date_str'], '%Y-%m-%d')
            pd_dt = datetime.strptime(p_fields['date_str'], '%Y-%m-%d')
            diff = abs((kd - pd_dt).days)
            if diff <= 1:
                score += 0.2
        except ValueError:
            pass

    # threshold (0.2)
    kt = k_fields.get('threshold')
    pt = p_fields.get('threshold')
    if kt is not None and pt is not None:
        if abs(kt - pt) / max(abs(kt), 1) < 0.01:
            score += 0.2

    return score


def _embedding_sim_sbert(text_a: str, text_b: str, model) -> float:
    vecs = model.encode([text_a, text_b], show_progress_bar=False)
    a, b = vecs[0], vecs[1]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(cos)


def _embedding_sim_fallback(text_a: str, text_b: str) -> float:
    # jaccard on words when sbert not available
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def embedding_similarity(text_a: str, text_b: str, model=None) -> float:
    if model is not None:
        return _embedding_sim_sbert(text_a, text_b, model)
    return _embedding_sim_fallback(text_a, text_b)


def fuzzy_match_score(text_a: str, text_b: str) -> float:
    if not HAS_FUZZ:
        # poor man's fuzzy: substring matching on parts
        parts_a = re.split(r'[-_ ]', text_a.lower())
        hits = sum(1 for p in parts_a if p and p in text_b.lower())
        return hits / max(len(parts_a), 1)
    return fuzz.token_set_ratio(text_a, text_b) / 100.0


def mock_llm_verify(ticker: str, desc: str, k_fields: Dict, p_fields: Dict) -> Tuple[bool, float]:
    # mock -- swap this for a real api call to gpt-4/claude
    rule_score = rule_based_match(k_fields, p_fields)

    if rule_score > 0.8:
        return (True, 0.95)
    elif rule_score >= 0.5:
        k_teams = set(t.lower() for t in (k_fields.get('teams') or []))
        p_teams = set(t.lower() for t in (p_fields.get('teams') or []))
        has_overlap = len(k_teams & p_teams) > 0
        if has_overlap:
            return (True, 0.7)
        else:
            return (False, 0.6)
    else:
        return (False, 0.9)


# ---- hybrid matcher ----
# supports both the configurable pipeline variant interface (for figures)
# and the original cascading-threshold interface

class HybridMatcher:
    def __init__(
        self,
        sbert_model: str = 'all-MiniLM-L6-v2',
        embed_thresh: float = 0.85,
        fuzzy_thresh: float = 0.80,
        llm_range: Tuple[float, float] = (0.70, 0.90),
        # pipeline variant toggles (used for ablation figures)
        use_regex: bool = True,
        use_embedding: bool = True,
        use_fuzzy: bool = True,
        use_llm: bool = True,
        threshold: float = 0.5,
    ):
        self.embed_thresh = embed_thresh
        self.fuzzy_thresh = fuzzy_thresh
        self.llm_range = llm_range
        self.use_regex = use_regex
        self.use_embedding = use_embedding
        self.use_fuzzy = use_fuzzy
        self.use_llm = use_llm
        self.threshold = threshold

        # weights for combining scores in ablation mode
        self.w_regex = 0.25
        self.w_embed = 0.35
        self.w_fuzzy = 0.15
        self.w_llm = 0.25

        # try loading the embedding model
        self.model = None
        if use_embedding and HAS_SBERT:
            try:
                self.model = SentenceTransformer(sbert_model)
            except Exception as e:
                print(f"couldn't load sbert model: {e}")
                self.model = None

    def score_combined(self, ticker, description):
        # returns a single float for threshold-based evaluation
        k_fields = extract_kalshi_fields(ticker)
        p_fields = extract_polymarket_fields(description)

        scores = {}
        total_weight = 0

        if self.use_regex and k_fields is not None:
            scores['regex'] = rule_based_match(k_fields, p_fields)
            total_weight += self.w_regex
        else:
            scores['regex'] = 0.0

        if self.use_embedding:
            scores['embedding'] = embedding_similarity(ticker, description, self.model)
            total_weight += self.w_embed
        else:
            scores['embedding'] = 0.0

        if self.use_fuzzy:
            scores['fuzzy'] = fuzzy_match_score(ticker, description)
            total_weight += self.w_fuzzy
        else:
            scores['fuzzy'] = 0.0

        if self.use_llm and k_fields is not None:
            is_match, conf = mock_llm_verify(ticker, description, k_fields, p_fields)
            scores['llm'] = conf if is_match else (1.0 - conf)
            total_weight += self.w_llm
        else:
            scores['llm'] = 0.0

        # weighted average
        combined = 0
        if self.use_regex:
            combined += self.w_regex * scores['regex']
        if self.use_embedding:
            combined += self.w_embed * scores['embedding']
        if self.use_fuzzy:
            combined += self.w_fuzzy * scores['fuzzy']
        if self.use_llm:
            combined += self.w_llm * scores['llm']

        if total_weight > 0:
            combined /= total_weight

        return combined

    def match(self, kalshi_ticker: str, poly_desc: str) -> MatchResult:
        t0 = time.time()

        # use combined scoring
        combined = self.score_combined(kalshi_ticker, poly_desc)
        is_match = combined >= self.threshold

        # determine which method contributed most
        if is_match:
            method = 'combined'
        else:
            method = 'rejected'

        elapsed = (time.time() - t0) * 1000
        return MatchResult(is_match, combined, method, elapsed)

    def match_batch(self, pairs: List[Tuple[str, str]]) -> List[MatchResult]:
        results = []
        for i, (ticker, desc) in enumerate(pairs):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"matching {i+1}/{len(pairs)}...")
            results.append(self.match(ticker, desc))
        print(f"done, matched {len(pairs)} pairs")
        return results


# ---- evaluation ----
# accepts benchmark records directly (list of dicts from generate_matching_benchmark)

def evaluate_matching(records, matcher, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 50)

    # get combined scores for all pairs
    all_scores = []
    all_labels = []
    all_categories = []
    for r in records:
        s = matcher.score_combined(r['kalshi_ticker'], r['poly_desc'])
        all_scores.append(s)
        all_labels.append(r['is_match'])
        all_categories.append(r['category'])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # precision/recall at each threshold
    precisions = []
    recalls = []
    f1s = []
    for t in thresholds:
        preds = all_scores >= t
        tp = np.sum(preds & all_labels)
        fp = np.sum(preds & ~all_labels)
        fn = np.sum(~preds & all_labels)

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # per-category f1 at best threshold
    best_t_idx = np.argmax(f1s)
    best_threshold = thresholds[best_t_idx]

    category_f1 = {}
    for cat in set(all_categories):
        mask = np.array([c == cat for c in all_categories])
        cat_scores = all_scores[mask]
        cat_labels = all_labels[mask]
        preds = cat_scores >= best_threshold
        tp = np.sum(preds & cat_labels)
        fp = np.sum(preds & ~cat_labels)
        fn = np.sum(~preds & cat_labels)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        category_f1[cat] = f1

    return {
        'thresholds': thresholds,
        'precisions': np.array(precisions),
        'recalls': np.array(recalls),
        'f1s': np.array(f1s),
        'best_threshold': best_threshold,
        'best_f1': f1s[best_t_idx],
        'category_f1': category_f1,
        'all_scores': all_scores,
        'all_labels': all_labels,
    }


# ---- latency benchmarking ----

def time_pipeline_stages(records, n_sample=100):
    # time each stage on a sample of pairs, return mean and std per stage
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(records), size=min(n_sample, len(records)), replace=False)
    sample = [records[i] for i in sample_idx]

    timings = {}

    # load sbert model once if available
    sbert_model = None
    if HAS_SBERT:
        try:
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            pass

    # regex stage
    times = []
    for _ in range(3):
        t0 = time.time()
        for r in sample:
            k = extract_kalshi_fields(r['kalshi_ticker'])
            p = extract_polymarket_fields(r['poly_desc'])
            if k is not None:
                rule_based_match(k, p)
        times.append((time.time() - t0) / len(sample) * 1000)
    timings['Regex'] = {'mean': np.mean(times), 'std': np.std(times)}

    # embedding stage
    times = []
    for _ in range(3):
        t0 = time.time()
        for r in sample:
            embedding_similarity(r['kalshi_ticker'], r['poly_desc'], sbert_model)
        times.append((time.time() - t0) / len(sample) * 1000)
    timings['SBERT Embed'] = {'mean': np.mean(times), 'std': np.std(times)}

    # fuzzy stage
    times = []
    for _ in range(3):
        t0 = time.time()
        for r in sample:
            fuzzy_match_score(r['kalshi_ticker'], r['poly_desc'])
        times.append((time.time() - t0) / len(sample) * 1000)
    timings['Rapidfuzz'] = {'mean': np.mean(times), 'std': np.std(times)}

    # mock llm stage
    times = []
    for _ in range(3):
        t0 = time.time()
        for r in sample:
            k = extract_kalshi_fields(r['kalshi_ticker'])
            p = extract_polymarket_fields(r['poly_desc'])
            if k is not None:
                mock_llm_verify(r['kalshi_ticker'], r['poly_desc'], k, p)
        times.append((time.time() - t0) / len(sample) * 1000)
    timings['LLM Verify'] = {'mean': np.mean(times), 'std': np.std(times)}

    # full pipeline
    matcher = HybridMatcher(threshold=0.5)
    times = []
    for _ in range(3):
        t0 = time.time()
        for r in sample:
            matcher.score_combined(r['kalshi_ticker'], r['poly_desc'])
        times.append((time.time() - t0) / len(sample) * 1000)
    timings['Full Pipeline'] = {'mean': np.mean(times), 'std': np.std(times)}

    return timings


if __name__ == '__main__':
    from simulate_markets import generate_matching_benchmark

    bench = generate_matching_benchmark(n_pairs=50, seed=0)
    matcher = HybridMatcher(threshold=0.4)
    result = evaluate_matching(bench, matcher)
    print(f"best f1: {result['best_f1']:.3f} at threshold {result['best_threshold']:.2f}")
    print(f"category f1: {result['category_f1']}")

    timings = time_pipeline_stages(bench, n_sample=20)
    for stage, t in timings.items():
        print(f"  {stage}: {t['mean']:.2f}ms (+/- {t['std']:.2f})")
