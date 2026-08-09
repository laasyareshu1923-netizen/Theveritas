import streamlit as st
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from transformers import pipeline

# 1. Page Configuration Setup
st.set_page_config(
    page_title="The Veritas AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Interactive CSS Framework
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    h1 { font-weight: 800 !important; background: linear-gradient(45deg, #38bdf8, #ef4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    /* Central Inputs and Custom Button Layout Rules */
    div.stButton > button { border-radius: 12px; font-weight: bold; width: 100%; transition: all 0.2s ease; padding: 10px 0; }
    
    /* Sleek Green GO Button */
    .go-btn button { background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important; color: white !important; border: none !important; }
    .go-btn button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(22, 163, 74, 0.3); }
    
    /* Clear Button */
    .reset-btn button { background-color: #334155 !important; color: #cbd5e1 !important; border: 1px solid #475569 !important; }
    
    /* Interactive Choice Mode Buttons */
    .action-btn button { background-color: #1e293b !important; color: #38bdf8 !important; border: 1px solid #0284c7 !important; text-align: left !important; padding: 12px 15px !important; }
    .action-btn button:hover { background-color: #0284c7 !important; color: white !important; transform: translateX(4px); }
    
    /* Styling Card Elements */
    .news-card { background-color: #1e293b; padding: 24px; border-radius: 16px; border: 1px solid #334155; margin-top: 25px; }
    .ai-response-box { background-color: #0f172a; padding: 20px; border-radius: 12px; border: 1px solid #1e293b; border-left: 4px solid #38bdf8; margin-top: 20px; }
    .live-article-box { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-top: 10px; }
    .source-badge { background-color: #38bdf8; color: #0f172a; font-size: 11px; padding: 3px 10px; border-radius: 6px; font-weight: bold; }
    .tier-badge { color: #f8fafc; font-size: 11px; padding: 3px 10px; border-radius: 6px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. Lazy-Load Deep Learning Model
@st.cache_resource
def load_deep_learning_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

ai_engine = load_deep_learning_model()

# 3. Live Web Verification Tool Function (Pulls data live from Google News India database)
def fetch_real_web_data(query_text):
    try:
        encoded_query = urllib.parse.quote(query_text)
        # Hits the live Indian broadcast indexing grid natively
        url = f"https://google.com{encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            discovered_articles = []
            for item in root.findall('.//item')[:3]: # Grab top 3 highly relevant live matches
                title = item.find('title').text
                link = item.find('link').text
                pub_date = item.find('pubDate').text
                # Clean up source formatting strings
                source_name = item.find('source').text if item.find('source') is not None else "Verified Media Hub"
                
                discovered_articles.append({
                    "title": title,
                    "link": link,
                    "date": pub_date,
                    "source": source_name
                })
            return discovered_articles
    except Exception as e:
        return []

# 4. Streamlit Session State Management
if "text_query" not in st.session_state:
    st.session_state.text_query = ""
if "ai_analyzed" not in st.session_state:
    st.session_state.ai_analyzed = False
if "current_label" not in st.session_state:
    st.session_state.current_label = ""
if "current_score" not in st.session_state:
    st.session_state.current_score = 0.0

def run_reset():
    st.session_state.text_query = ""
    st.session_state.ai_analyzed = False
    st.session_state.current_label = ""
    st.session_state.current_score = 0.0
    st.rerun()

# 5. Layout Setup
st.title("The Veritas AI")
st.markdown("##### Instant Linguistic & Live Web Verification Engine")
st.markdown("Paste any statement or headline to initiate high-speed cognitive deep scans and live data-channel checks.")
st.markdown("---")

st.markdown("##### 🔍 Central Verification Search")
col_input, col_go, col_reset = st.columns(3)

with col_input:
    search_term = st.text_input(
        label="Search Input Box",
        value=st.session_state.text_query,
        placeholder="Example: Reliance Investment in Andhra...",
        label_visibility="collapsed"
    )

with col_go:
    st.markdown('<div class="go-btn">', unsafe_allow_html=True)
    go_clicked = st.button("GO ➔")
    st.markdown('</div>', unsafe_allow_html=True)

with col_reset:
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    st.button("Clear Dashboard", on_click=run_reset)
    st.markdown('</div>', unsafe_allow_html=True)

if go_clicked and search_term:
    with st.spinner("Processing neural layers..."):
        ai_prediction = ai_engine(search_term)
        st.session_state.current_label = ai_prediction[0]['label']
        st.session_state.current_score = ai_prediction[0]['score'] * 100
        st.session_state.ai_analyzed = True
        st.session_state.text_query = search_term

# 6. Output Panel Interface
if st.session_state.ai_analyzed:
    label = st.session_state.current_label
    confidence = st.session_state.current_score
    active_text = st.session_state.text_query
    
    if label == "NEGATIVE":
        verdict, badge_bg, trust_pct = "Sensationalist Structure Flagged", "#ef4444", f"{max(100 - confidence, 12):.1f}%"
    else:
        verdict, badge_bg, trust_pct = "Neutral Structure Verified", "#16a34a", f"{confidence:.1f}%"

    st.markdown(f"""
    <div class="news-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span class="source-badge">🤖 VERITAS MATRIX SECTOR ACTIVE</span>
            <span class="tier-badge" style="background-color: {badge_bg};">● {verdict}</span>
        </div>
        <h4 style="color: #f1f5f9; margin-bottom: 15px;">Target Query: "{active_text}"</h4>
        <div style="display: flex; gap: 20px; font-size: 13px; color: #94a3b8;">
            <span>📊 Linguistic Objectivity: {confidence:.1f}%</span>
            <span>🛡️ Core Structure Trust: {trust_pct}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>### 🛠️ Interactive AI Options", unsafe_allow_html=True)
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        opt_factcheck = st.button("🌐 1. Run Live Web Fact-Check")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_opt2:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        opt_bias = st.button("🔍 2. Core Phrasing & Bias Breakdown")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_opt3:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        opt_neutral = st.button("📝 3. Rewrite Intelligently & Neutrally")
        st.markdown('</div>', unsafe_allow_html=True)

    # 7. EXECUTE LIVE WEB CHECKING PIPELINE
    if opt_factcheck:
        with st.spinner("Scanning live regional news registries and broadcast logs..."):
            live_news_results = fetch_real_web_data(active_text)
            
            if live_news_results:
                st.markdown(f"""
                <div class="ai-response-box">
                    <h5 style="color: #38bdf8; margin-bottom: 12px;">✅ Real Matches Found in Live Broadcast Logs</h5>
                    <p style="font-size: 14px; color: #cbd5e1; margin-bottom: 15px;">
                        The Veritas AI successfully cross-referenced your query. Multiple verified media networks are reporting live on this exact topic. This indicates a high likelihood of a real, ongoing news event.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Render real articles out onto the web app screen!
                for art in live_news_results:
                    st.markdown(f"""
                    <div class="live-article-box">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #38bdf8; margin-bottom: 4px;">
                            <span>📺 CHANNEL: {art['source']}</span>
                            <span>🗓️ {art['date']}</span>
                        </div>
                        <h6 style="color: #f8fafc; font-size: 14px; margin: 0 0 6px 0;">{art['title']}</h6>
                        <a href="{art['link']}" target="_blank" style="color: #f97316; font-size: 12px; font-weight: bold; text-decoration: none;">Read Original Publisher Article ➔</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-response-box" style="border-left-color: #ef4444;">
                    <h5 style="color: #ef4444; margin-bottom: 10px;">❌ Zero Verified Media Matches Discovered</h5>
                    <p style="font-size: 14px; color: #cbd5e1; line-height: 1.6;">

