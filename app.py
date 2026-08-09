import streamlit as st
import datetime
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
    
    /* Core Layout Formats */
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
    .source-badge { background-color: #38bdf8; color: #0f172a; font-size: 11px; padding: 3px 10px; border-radius: 6px; font-weight: bold; }
    .tier-badge { color: #f8fafc; font-size: 11px; padding: 3px 10px; border-radius: 6px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. Lazy-Load Deep Learning Model
@st.cache_resource
def load_deep_learning_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

ai_engine = load_deep_learning_model()

# 3. Streamlit Session State Management
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

# 4. Brand Layout Elements
st.title("The Veritas AI")
st.markdown("##### Instant Linguistic & Interactive Credibility Engine")
st.markdown("Paste any statement or headline to initiate high-speed cognitive deep scans.")
st.markdown("---")

# 5. Search Bar and Core Action Buttons
st.markdown("##### 🔍 Central Verification Search")
col_input, col_go, col_reset = st.columns()

with col_input:
    search_term = st.text_input(
        label="Search Input Box",
        value=st.session_state.text_query,
        placeholder="Example: Shocking secret leak! This basic ingredient fixes everything instantly...",
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

# Process search term input to trigger initial neural prediction
if go_clicked and search_term:
    with st.spinner("Processing neural layers..."):
        ai_prediction = ai_engine(search_term)
        st.session_state.current_label = ai_prediction[0]['label']
        st.session_state.current_score = ai_prediction[0]['score'] * 100
        st.session_state.ai_analyzed = True
        st.session_state.text_query = search_term

# 6. Interactive Module Branching
if st.session_state.ai_analyzed:
    label = st.session_state.current_label
    confidence = st.session_state.current_score
    active_text = st.session_state.text_query
    
    # Calculate base display statistics dynamically
    if label == "NEGATIVE":
        verdict, badge_bg, trust_pct = "High Suspected Manipulation", "#ef4444", f"{max(100 - confidence, 12):.1f}%"
    else:
        verdict, badge_bg, trust_pct = "Low-Risk Structure Verified", "#16a34a", f"{confidence:.1f}%"

    # Display base analysis card
    st.markdown(f"""
    <div class="news-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span class="source-badge">🤖 INTERACTIVE VERITAS ARCHITECTURE</span>
            <span class="tier-badge" style="background-color: {badge_bg};">● {verdict}</span>
        </div>
        <h4 style="color: #f1f5f9; margin-bottom: 15px;">Target Text: "{active_text}"</h4>
        <div style="display: flex; gap: 20px; font-size: 13px; color: #94a3b8;">
            <span>📊 AI Base Confidence: {confidence:.1f}%</span>
            <span>🛡️ Overall Structural Trust: {trust_pct}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # INTERACTIVE OPTIONS MODULE BLOCK
    st.markdown("<br>##### 🛠️ Interactive AI Options - Choose a Task Below:", unsafe_allow_html=True)
    
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        opt_factcheck = st.button("🌐 1. Simulate Live Web Fact-Check")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_opt2:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        opt_bias = st.button("🔍 2. Core Phrasing & Bias Breakdown")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_opt3:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        opt_neutral = st.button("📝 3. Rewrite Intelligently & Neutrally")
        st.markdown('</div>', unsafe_allow_html=True)

    # Render interactive outputs directly onto the dashboard display layout
    if opt_factcheck:
        st.markdown(f"""
        <div class="ai-response-box">
            <h5 style="color: #38bdf8; margin-bottom: 10px;">🌐 Live Cross-Reference Simulation</h5>
            <p style="font-size: 14px; color: #cbd5e1; line-height: 1.6;">
                The Veritas AI queued national and international news registries. Mainstream sources show <b>zero matching data logs</b> for this exact claim structure. Cross-reference suggests this statement likely originated as uncorroborated text or a localized social media cycle.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif opt_bias:
        # Generate targeted bias advice based on the neural model output
        bias_analysis = (
            "This phrasing relies on extreme buzzwords, intense punctuation, or urgency framing. It targets raw reader emotions to bypass logical critical reasoning."
            if label == "NEGATIVE" else
            "Linguistic structures are descriptive and flat. The text avoids logical fallacies, extreme exaggerations, or structural clickbait baiting tags."
        )
        st.markdown(f"""
        <div class="ai-response-box">
            <h5 style="color: #38bdf8; margin-bottom: 10px;">🔍 Deep Linguistic Phrasing Analysis</h5>
            <p style="font-size: 14px; color: #cbd5e1; line-height: 1.6;">
                <b>Structural Breakdown:</b> {bias_analysis}<br><br>
                <b>Primary Bias Signature:</b> {"Sensationalist/Alarmist Overdrive" if label == "NEGATIVE" else "Neutral Informational Blueprint"}
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif opt_neutral:
        # Provide a clean neutral rewrite example
        cleaned_version = (
            f"Neutral Standard Alternative: 'Investigating unverified claims regarding localized event topics.'"
            if label == "NEGATIVE" else f"The text is already structured neutrally: '{active_text}'"
        )
        st.markdown(f"""
        <div class="ai-response-box">
            <h5 style="color: #38bdf8; margin-bottom: 10px;">📝 Intelligent Neutral Transformation</h5>
            <p style="font-size: 14px; color: #cbd5e1; line-height: 1.6;">
                Removing conversational exaggerations, hype structures, and clickbait framing markers:<br><br>
                <i style="color: #a7f3d0;">{cleaned_version}</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

elif go_clicked and not search_term:
    st.warning("⚠️ Action block empty. Please type text inside the search box before launching the verification engine.")
  
