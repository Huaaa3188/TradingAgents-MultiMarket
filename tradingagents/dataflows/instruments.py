import re
from enum import Enum


class InstrumentType(str, Enum):
    EQUITY = "equity"
    FUND = "fund"
    CRYPTO = "crypto"
    UNKNOWN = "unknown"


class MarketType(str, Enum):
    US = "us"
    CN_A = "cn_a"
    CN_FUND = "cn_fund"
    HK = "hk"
    JP = "jp"
    CRYPTO = "crypto"
    OTHER = "other"


CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")
CN_A_SUFFIXES = (".SH", ".SZ", ".BJ")

# 场内基金（ETF/LOF/REIT）代码段：沪市 5xxxxx；深市 159xxx / 16xxxx / 18xxxx
CN_A_FUND_SH_PREFIXES = ("5",)
CN_A_FUND_SZ_PREFIXES = ("159", "16", "18")

# A 股股票代码段。使用精确段位而非宽前缀，避免把同为 6 位数字的场外基金
# （如 005827 易方达蓝筹精选、110011 易方达中小盘）误判为深市股票或未识别市场：
#   沪市主板：600/601/603/605；科创板：688
#   深市主板：000/001/002/003；创业板：300/301
#   北交所（含新三板）：8xxxxx、920xxx、4xxxxx
CN_A_EQUITY_SH_PREFIXES = ("600", "601", "603", "605", "688")
CN_A_EQUITY_SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")
CN_A_EQUITY_BJ_PREFIXES = ("8", "920", "4")

_SIX_DIGIT_RE = re.compile(r"^\d{6}$")


def normalize_ticker_symbol(ticker: str) -> str:
    """Normalize user input while preserving or adding known exchange suffixes.

    Bare six-digit China codes are classified purely by prefix syntax:
    exchange-traded funds and equities get their exchange suffix
    (``.SH``/``.SZ``/``.BJ``); every remaining six-digit code is treated as a
    China OTC fund code and kept bare (Tiantian Fund / Eastmoney code). No
    network is involved here — OTC confirmation happens in the cached data
    layer, so classification stays deterministic and synchronous.
    """
    normalized = ticker.strip().upper()
    if not _SIX_DIGIT_RE.fullmatch(normalized):
        return normalized

    kind, exchange = _classify_cn_six_digit(normalized)
    if kind == "otc_fund":
        return normalized
    return f"{normalized}.{exchange}"


def detect_market_type(ticker: str) -> MarketType:
    normalized = normalize_ticker_symbol(ticker)
    if normalized.endswith(CRYPTO_SUFFIXES):
        return MarketType.CRYPTO
    code, _, suffix = normalized.partition(".")
    if _SIX_DIGIT_RE.fullmatch(code) and _classify_cn_six_digit(code)[0] == "otc_fund":
        return MarketType.CN_FUND
    if normalized.endswith(CN_A_SUFFIXES):
        return MarketType.CN_A
    if normalized.endswith(".HK"):
        return MarketType.HK
    if normalized.endswith(".T"):
        return MarketType.JP
    if _SIX_DIGIT_RE.fullmatch(normalized):
        return MarketType.OTHER
    if "." not in normalized and "-" not in normalized and normalized:
        return MarketType.US
    return MarketType.OTHER


def detect_instrument_type(ticker: str) -> InstrumentType:
    normalized = normalize_ticker_symbol(ticker)
    if normalized.endswith(CRYPTO_SUFFIXES):
        return InstrumentType.CRYPTO
    code, _, suffix = normalized.partition(".")
    if _SIX_DIGIT_RE.fullmatch(code):
        kind, _ = _classify_cn_six_digit(code)
        if kind in ("fund", "otc_fund"):
            return InstrumentType.FUND
        if kind == "equity":
            return InstrumentType.EQUITY
    if normalized and not _SIX_DIGIT_RE.fullmatch(normalized):
        return InstrumentType.EQUITY
    return InstrumentType.UNKNOWN


def is_cn_a_ticker(ticker: str) -> bool:
    return detect_market_type(ticker) == MarketType.CN_A


def is_cn_a_fund(ticker: str) -> bool:
    return detect_instrument_type(ticker) == InstrumentType.FUND and is_cn_a_ticker(ticker)


def is_cn_otc_fund(ticker: str) -> bool:
    return detect_market_type(ticker) == MarketType.CN_FUND and detect_instrument_type(ticker) == InstrumentType.FUND


def to_akshare_symbol(ticker: str) -> str:
    """Return the pure numeric symbol expected by AkShare's A-share APIs."""
    normalized = normalize_ticker_symbol(ticker)
    if normalized.endswith(CN_A_SUFFIXES):
        return normalized.split(".", 1)[0]
    return normalized


def _classify_cn_six_digit(code: str) -> tuple[str, str | None]:
    """Classify a bare six-digit China code by prefix syntax alone.

    Returns ``(kind, exchange)`` where kind is one of ``"fund"``
    (exchange-traded fund), ``"equity"``, or ``"otc_fund"`` (China OTC mutual
    fund); exchange is ``"SH"``/``"SZ"``/``"BJ"`` for listed instruments and
    ``None`` for OTC funds. Deterministic and synchronous by design — no
    network. Residual codes (e.g. B-shares ``200xxx``/``900xxx``) default to
    the OTC-fund candidate and are rejected later by the cached data layer,
    which is safer than mislabelling them as equities.
    """
    if code.startswith(CN_A_FUND_SH_PREFIXES + CN_A_FUND_SZ_PREFIXES):
        return "fund", _cn_a_exchange_for_fund(code)
    if code.startswith(CN_A_EQUITY_SH_PREFIXES):
        return "equity", "SH"
    if code.startswith(CN_A_EQUITY_SZ_PREFIXES):
        return "equity", "SZ"
    if code.startswith(CN_A_EQUITY_BJ_PREFIXES):
        return "equity", "BJ"
    return "otc_fund", None


def _is_cn_a_fund_code(code: str) -> bool:
    return _classify_cn_six_digit(code)[0] == "fund"


def _is_cn_otc_fund_code(code: str) -> bool:
    return _classify_cn_six_digit(code)[0] == "otc_fund"


def _cn_a_exchange_for_fund(code: str) -> str:
    if code.startswith(CN_A_FUND_SH_PREFIXES):
        return "SH"
    return "SZ"
