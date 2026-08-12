from __future__ import annotations

BG = "#0B1017"
PANEL = "#131B26"
PANEL_2 = "#18222F"
BORDER = "#243141"
TEXT = "#E6EDF6"
MUTED = "#8A9BAF"
ACCENT = "#4C8DFF"
POS = "#2FD4A7"
NEG = "#FF5C7A"
WARN = "#FFB454"

SERIES = ["#4C8DFF", "#FF9E5E", "#2FD4A7", "#B98CFF", "#FF5C7A", "#6EC8FF", "#F7D65E"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, Segoe UI, system-ui, sans-serif", size=12),
    margin=dict(l=10, r=10, t=44, b=10),
    hoverlabel=dict(bgcolor=PANEL_2, bordercolor=BORDER, font=dict(color=TEXT, size=12)),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", orientation="h",
                yanchor="bottom", y=1.0, xanchor="right", x=1),
    colorway=SERIES,
)

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
  .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
  #MainMenu, footer, header { visibility: hidden; }

  .hero {
    background: linear-gradient(135deg, #16233A 0%, #101724 55%, #0B1017 100%);
    border: 1px solid #243141; border-radius: 16px;
    padding: 22px 26px; margin-bottom: 18px;
  }
  .hero h1 {
    font-size: 1.65rem; font-weight: 700; margin: 0 0 6px 0;
    color: #E6EDF6; letter-spacing: -0.02em;
  }
  .hero p { color: #8A9BAF; margin: 0; font-size: 0.94rem; line-height: 1.5; }
  .pill {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; margin-right: 8px; margin-top: 12px;
    border: 1px solid #2B3A4D; color: #9FB3C8; background: rgba(76,141,255,0.08);
  }
  .pill.ok { color: #2FD4A7; border-color: rgba(47,212,167,0.35); background: rgba(47,212,167,0.08); }

  .kpi {
    background: #131B26; border: 1px solid #243141; border-radius: 14px;
    padding: 16px 18px; height: 100%;
    transition: border-color .18s ease, transform .18s ease;
  }
  .kpi:hover { border-color: #33475F; transform: translateY(-1px); }
  .kpi .label {
    color: #8A9BAF; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.07em; text-transform: uppercase;
  }
  .kpi .value {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 1.6rem; font-weight: 600; margin-top: 6px; color: #E6EDF6; line-height: 1.1;
  }
  .kpi .value.pos { color: #2FD4A7; }
  .kpi .value.neg { color: #FF5C7A; }
  .kpi .sub { color: #6C7E93; font-size: 0.76rem; margin-top: 5px; }

  .section-title {
    font-size: 1.02rem; font-weight: 650; color: #E6EDF6;
    margin: 26px 0 4px 0; letter-spacing: -0.01em;
  }
  .section-note { color: #8A9BAF; font-size: 0.86rem; margin-bottom: 12px; line-height: 1.55; }

  .callout {
    border-left: 3px solid #4C8DFF; background: rgba(76,141,255,0.06);
    padding: 12px 16px; border-radius: 0 10px 10px 0; margin: 10px 0 16px 0;
    color: #B9C7D6; font-size: 0.87rem; line-height: 1.6;
  }
  .callout.warn { border-left-color: #FFB454; background: rgba(255,180,84,0.06); }
  .callout.good { border-left-color: #2FD4A7; background: rgba(47,212,167,0.06); }

  .stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #243141; }
  .stTabs [data-baseweb="tab"] {
    height: 42px; background: transparent; border-radius: 8px 8px 0 0;
    color: #8A9BAF; font-weight: 550; font-size: 0.9rem; padding: 0 16px;
  }
  .stTabs [aria-selected="true"] { color: #E6EDF6 !important; background: #131B26 !important; }

  section[data-testid="stSidebar"] { background: #0E141D; border-right: 1px solid #243141; }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

  [data-testid="stDataFrame"] { border: 1px solid #243141; border-radius: 10px; }
  .stDownloadButton button, .stButton button {
    border: 1px solid #2B3A4D; background: #18222F; color: #E6EDF6;
    border-radius: 9px; font-weight: 550;
  }
  .stDownloadButton button:hover, .stButton button:hover { border-color: #4C8DFF; color: #4C8DFF; }
  .mono { font-family: 'JetBrains Mono', monospace; }
</style>
"""


def kpi_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = f"value {tone}".strip()
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="{cls}">{value}</div>{sub_html}</div>'
    )
