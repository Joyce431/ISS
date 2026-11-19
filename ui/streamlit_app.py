"""DeepGuard AI - Streamlit Web Interface"""
import streamlit as st
import requests
import json
import time
import os
import glob
import pandas as pd
import plotly.express as px
from datetime import datetime

API_URL = "http://localhost:8000/api/v1"

REPORTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "reports")
)
os.makedirs(REPORTS_DIR, exist_ok=True)

st.set_page_config(
    page_title="DeepGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Light-mode CSS ----------------
st.markdown(
    """
<style>
    .main-header { font-size: 3em; color: #1a73e8; text-align: center; margin-bottom: 0; }
    .sub-header  { text-align: center; color: #666; margin-bottom: 30px; }

    .risk-critical { background: #dc3545; color: white; padding: 15px; border-radius: 8px; }
    .risk-high     { background: #fd7e14; color: white; padding: 15px; border-radius: 8px; }
    .risk-medium   { background: #ffc107; color: black; padding: 15px; border-radius: 8px; }
    .risk-low      { background: #20c997; color: white; padding: 15px; border-radius: 8px; }
    .risk-minimal  { background: #28a745; color: white; padding: 15px; border-radius: 8px; }

    .finding-box {
        background-color: #fff3cd; color: #856404; padding: 12px;
        border-left: 4px solid #ffeeba; border-radius: 4px; margin: 5px 0;
    }
    .rec-box {
        background-color: #d4edda; color: #155724; padding: 12px;
        border-left: 4px solid #c3e6cb; border-radius: 4px; margin: 5px 0;
    }
    .e2ee-box {
        background-color: #e7f3ff; color: #0b4f8a; padding: 12px;
        border-left: 4px solid #1a73e8; border-radius: 4px; margin: 5px 0;
        font-family: monospace; word-break: break-all;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN
# ============================================================
def main():
    # Header
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        try:
            st.image("lp_logo.jpeg", width=150)
        except Exception:
            pass
    with col2:
        st.markdown(
            '<h1 class="main-header" style="margin-top:-5px;">🛡️ DeepGuard AI</h1>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p class="sub-header">AI-Generated Phishing Email & Deepfake Media Detector</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        try:
            st.image("lp_logo.jpeg", width=150)
        except Exception:
            pass

        st.title("Navigation")
        page = st.radio(
            "Go to",
            [
                "🔍 Analyze File",
                "📧 Analyze Email Text",
                "🔒 Secure Chat",
                "📊 Dashboard",
                "ℹ️ About",
            ],
        )

        # Save button only appears after a scan
        if "last_report" in st.session_state:
            st.divider()
            if st.button("💾 Save Last Scan to Dashboard",
                         use_container_width=True, type="primary"):
                try:
                    report = st.session_state["last_report"]
                    save_path = os.path.join(
                        REPORTS_DIR, f"{report['analysis_id']}.json"
                    )
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2, default=str)
                    st.success("Saved successfully!")
                except Exception as e:
                    st.error(f"Failed to save: {e}")

        st.divider()
        st.subheader("System Status")
        try:
            health = requests.get(f"{API_URL}/health", timeout=5).json()
            st.success(f"✅ API Online v{health['version']}")
            st.write("**Models:**")
            for model, loaded in health["models_loaded"].items():
                icon = "✅" if loaded else "⚠️"
                st.write(f"{icon} {model.replace('_', ' ').title()}")
        except Exception:
            st.error("❌ API Offline")

    # Routing
    if page == "🔍 Analyze File":
        analyze_file_page()
    elif page == "📧 Analyze Email Text":
        analyze_email_page()
    elif page == "🔒 Secure Chat":
        secure_chat_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    else:
        about_page()


# ============================================================
# FILE ANALYSIS
# ============================================================
def analyze_file_page():
    st.header("🔍 File Analysis")
    st.write("Upload an email, image, audio, or video file for deepfake/phishing analysis.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=[
            "eml", "msg", "txt",
            "jpg", "jpeg", "png", "webp",
            "mp3", "wav", "flac", "ogg",
            "mp4", "avi", "mov", "mkv",
        ],
    )

    analysis_type = st.selectbox(
        "Analysis Type (optional override)",
        ["Auto-detect", "email", "image", "audio", "video"],
    )

    if uploaded_file and st.button("🚀 Analyze", type="primary", use_container_width=True):
        with st.spinner("Analyzing file..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {}
            if analysis_type != "Auto-detect":
                data["analysis_type"] = analysis_type

            try:
                start = time.time()
                response = requests.post(f"{API_URL}/analyze", files=files, data=data)
                elapsed = time.time() - start

                if response.status_code == 200:
                    report = response.json()
                    display_results(report, elapsed)
                else:
                    st.error(
                        f"Analysis failed: {response.json().get('detail', 'Unknown error')}"
                    )
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# EMAIL TEXT ANALYSIS
# ============================================================
def analyze_email_page():
    st.header("📧 Email Text Analysis")
    st.write("Paste email content directly for phishing analysis.")

    col1, col2 = st.columns(2)
    with col1:
        sender = st.text_input("Sender (optional)", placeholder="ceo@company.com")
        subject = st.text_input(
            "Subject (optional)", placeholder="Urgent: Wire Transfer Required"
        )
    with col2:
        st.write("")
        st.write("")
        st.info("💡 Tip: Include the full email body for best results")

    body = st.text_area(
        "Email Body", height=300, placeholder="Paste email content here..."
    )

    if st.button("🚀 Analyze Email", type="primary", use_container_width=True):
        if not body:
            st.warning("Please enter email content")
            return

        with st.spinner("Analyzing email..."):
            try:
                response = requests.post(
                    f"{API_URL}/analyze/email/text",
                    data={"body": body, "subject": subject, "sender": sender},
                )
                if response.status_code == 200:
                    display_email_results(response.json())
                else:
                    st.error("Analysis failed")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# FILE RESULTS
# ============================================================
def display_results(report: dict, elapsed: float):
    st.session_state["last_report"] = report

    overall = report.get("overall_risk", {})
    risk_level = overall.get("level", "UNKNOWN")
    risk_score = overall.get("score", 0)

    st.markdown(
        f'<div class="risk-{risk_level.lower()}">'
        f"<h2>{risk_level} RISK — {risk_score:.1%}</h2>"
        f'<p>{"⚠️ THREAT DETECTED" if overall.get("is_threat") else "✅ No threat detected"}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Score", f"{risk_score:.1%}")
    col2.metric("Confidence", f"{overall.get('confidence', 0):.1%}")
    col3.metric(
        "Processing Time",
        f"{report.get('performance', {}).get('processing_time_seconds', 0):.2f}s",
    )
    col4.metric("File Size", f"{report.get('file_info', {}).get('size_mb', 0)} MB")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Findings", "🔬 Details", "📁 File Info", "📥 Export"]
    )
    detection = report.get("detection_result", {})

    with tab1:
        st.subheader("Indicators Found")
        findings = detection.get("findings", [])
        if findings:
            for f in findings:
                st.markdown(
                    f'<div class="finding-box">⚠️ {f}</div>', unsafe_allow_html=True
                )
        else:
            st.success("No suspicious indicators detected")

        st.subheader("Recommendations")
        for r in detection.get("recommendations", []):
            st.markdown(
                f'<div class="rec-box">✅ {r}</div>', unsafe_allow_html=True
            )

    with tab2:
        st.subheader("Technical Details")
        for name, data in detection.get("components", {}).items():
            with st.expander(f"🔧 {name.replace('_', ' ').title()}"):
                st.json(data)

    with tab3:
        st.subheader("File Information")
        st.json(report.get("file_info", {}))
        st.subheader("Integrity Verification")
        st.json(report.get("integrity", {}))

    with tab4:
        st.download_button(
            "📥 Download JSON Report",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"report_{report.get('analysis_id', 'unknown')}.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# EMAIL RESULTS
# ============================================================
def display_email_results(result: dict):
    risk_score = result.get("risk_score", 0)
    risk_level = result.get("risk_level", "UNKNOWN")

    st.markdown(
        f'<div class="risk-{risk_level.lower()}">'
        f"<h2>{risk_level} RISK — {risk_score:.1%}</h2>"
        f'<p>{"⚠️ PHISHING DETECTED" if result.get("is_phishing") else "✅ Likely legitimate"}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{risk_score:.1%}")
    col2.metric("Confidence", f"{result.get('confidence', 0):.1%}")
    col3.metric("Risk Level", risk_level)

    st.subheader("Findings")
    components = result.get("components", {})

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Urgency Keywords:**")
        urgency = components.get("rule_analysis", {}).get("urgency_keywords", [])
        if urgency:
            for kw in urgency:
                st.markdown(f"- 🔴 {kw}")
        else:
            st.write("None found")

        st.write("**Credential Keywords:**")
        creds = components.get("rule_analysis", {}).get("credential_keywords", [])
        if creds:
            for kw in creds:
                st.markdown(f"- 🔴 {kw}")
        else:
            st.write("None found")

    with col2:
        st.write("**Payment Keywords:**")
        payments = components.get("rule_analysis", {}).get("payment_keywords", [])
        if payments:
            for kw in payments:
                st.markdown(f"- 🔴 {kw}")
        else:
            st.write("None found")

        st.write("**URL Analysis:**")
        url_data = components.get("url_analysis", {})
        st.write(f"URLs found: {url_data.get('urls_found', 0)}")
        for finding in url_data.get("findings", []):
            st.warning(f"URL: {finding['url']}")
            for ind in finding["indicators"]:
                st.write(f"  - {ind}")

    st.subheader("Recommendations")
    for r in result.get("recommendations", []):
        st.markdown(f'<div class="rec-box">✅ {r}</div>', unsafe_allow_html=True)


# ============================================================
# SECURE CHAT (E2EE)
# ============================================================
def secure_chat_page():
    st.header("🔒 End-to-End Encrypted Chat")
    st.write("Chat between two users. Try opening this in two different browser tabs!")
    
    # --- 1. USER LOGIN (SESSION ROLE) ---
    col_role1, col_role2 = st.columns(2)
    with col_role1:
        current_user = st.selectbox("Who are you?", ["Kananelo", "Tumo"])
    with col_role2:
        if st.button("Clear Chat History"):
            requests.delete(f"{API_URL}/chat/clear")
            st.rerun()

    st.divider()

    # --- 2. SENDING A MESSAGE ---
    st.subheader(f"Send a message as {current_user}")
    recipient = "Tumo" if current_user == "Kananelo" else "Kananelo"
    
    with st.form(key="chat_form", clear_on_submit=True):
        message = st.text_input("Type your message:", placeholder="Enter message...")
        submit = st.form_submit_button("🔐 Encrypt & Send", type="primary", use_container_width=True)
        
        if submit and message:
            # Send to API
            try:
                res = requests.post(
                    f"{API_URL}/chat/send", 
                    data={"sender": current_user, "recipient": recipient, "message": message}
                )
                if res.status_code == 200:
                    st.success(f"Sent securely to {recipient}!")
                    st.code(res.json()["encrypted_on_wire"], language="text")
                    st.caption("↑ This is what the server sees.")
                else:
                    st.error("Failed to send.")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()

    # --- 3. RECEIVING MESSAGES ---
    st.subheader(f"📥 {current_user}'s Inbox")
    
    if st.button("🔄 Refresh Inbox"):
        st.rerun()
        
    try:
        res = requests.get(f"{API_URL}/chat/inbox/{current_user}")
        if res.status_code == 200:
            inbox = res.json()
            if inbox["message_count"] == 0:
                st.info("No messages yet. Send one from the other user!")
            else:
                for msg in inbox["messages"]:
                    st.markdown(
                        f"""
                        <div style="background-color:#e7f3ff; padding:15px; border-radius:8px; margin-bottom:10px; border-left:4px solid #1a73e8;">
                            <strong>From: {msg['from']}</strong> <span style="float:right; color:gray; font-size:0.9em;">{msg['time']}</span><br><br>
                            <strong>🔓 Decrypted Text:</strong> {msg['decrypted_text']}<br>
                            <div style="font-size:0.8em; color:#555; margin-top:8px; font-family:monospace; word-break:break-all;">
                                <strong>Wire Payload:</strong> {msg['encrypted_on_wire']}
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
    except Exception as e:
        st.error(f"Failed to load inbox: {e}")

# ============================================================
# DASHBOARD
# ============================================================
def dashboard_page():
    st.header("📊 Security Dashboard")
    st.write("This dashboard aggregates all files analyzed and saved during this session.")

    report_files = glob.glob(os.path.join(REPORTS_DIR, "*.json"))

    if not report_files:
        st.info(
            "No data yet. Go to **Analyze File**, run a scan, and click "
            "'Save Last Scan to Dashboard' in the sidebar to populate this view."
        )
        return

    data = []
    for file in report_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                r = json.load(f)
            if "overall_risk" in r and "file_info" in r:
                data.append(
                    {
                        "Date": datetime.fromisoformat(
                            r.get("timestamp", datetime.now().isoformat())
                        ).strftime("%H:%M:%S"),
                        "File Name": r["file_info"].get("name", "Unknown"),
                        "Type": r["file_info"].get("type", "unknown").upper(),
                        "Risk Level": r["overall_risk"].get("level", "UNKNOWN"),
                        "Risk Score": round(
                            r["overall_risk"].get("score", 0) * 100, 1
                        ),
                        "Time (sec)": r.get("performance", {}).get(
                            "processing_time_seconds", 0
                        ),
                        "Is Threat": r["overall_risk"].get("is_threat", False),
                    }
                )
        except Exception:
            pass

    if not data:
        st.warning("No valid reports found.")
        return

    df = pd.DataFrame(data)

    # KPI cards
    total_scans = len(df)
    threats_found = len(df[df["Is Threat"] == True])
    avg_time = round(df["Time (sec)"].mean(), 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Scans", total_scans)
    col2.metric(
        "Threats Detected",
        threats_found,
        delta="Action Needed" if threats_found > 0 else "All Clear",
        delta_color="inverse" if threats_found > 0 else "normal",
    )
    col3.metric("Avg Processing Time", f"{avg_time}s")

    st.divider()

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Risk Levels Distribution")
        fig1 = px.pie(
            df,
            names="Risk Level",
            color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#dc3545",
                "HIGH": "#fd7e14",
                "MEDIUM": "#ffc107",
                "LOW": "#20c997",
                "MINIMAL": "#28a745",
                "UNKNOWN": "#6c757d",
            },
            hole=0.4,
        )
        fig1.update_layout(
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            font=dict(color="#31333F", size=14),
            margin=dict(t=30, b=30, l=30, r=30),
            legend=dict(font=dict(color="#31333F")),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.subheader("Processing Time by Media Type")
        fig2 = px.bar(
            df,
            x="Type",
            y="Time (sec)",
            color="Type",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text="Time (sec)",
        )
        fig2.update_traces(
            texttemplate="%{text}s",
            textposition="outside",
            textfont=dict(color="#31333F", size=14),
        )
        fig2.update_layout(
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            font=dict(color="#31333F", size=14),
            xaxis=dict(
                title_font=dict(color="#31333F"),
                tickfont=dict(color="#31333F"),
            ),
            yaxis=dict(
                title_font=dict(color="#31333F"),
                tickfont=dict(color="#31333F"),
            ),
            margin=dict(t=30, b=30, l=30, r=30),
            legend=dict(font=dict(color="#31333F")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Recent Scan History")
    st.dataframe(
        df[
            ["Date", "File Name", "Type", "Risk Level", "Risk Score", "Time (sec)"]
        ].sort_values(by="Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ABOUT
# ============================================================
def about_page():
    st.header("ℹ️ About DeepGuard AI")
    st.markdown(
        """
    ### AI-Generated Phishing Email & Deepfake Media Detector

    **Version:** 1.0.0

    DeepGuard AI is a comprehensive defensive tool designed to detect:
    - 🎣 **Phishing emails** (including AI-generated text)
    - 🖼️ **Deepfake images** (face swaps, AI-generated)
    - 🔊 **Deepfake audio** (voice cloning, synthesis)
    - 🎥 **Deepfake videos** (face swaps, lip sync manipulation)

    ### Detection Models

    | Media Type | AI Model | Approach |
    |---|---|---|
    | Text/Email | DistilBERT + Rules | Transformer classification + URL analysis |
    | Image | EfficientNet-B4 + ELA | CNN classifier + forensic analysis |
    | Audio | Wav2Vec2 + Spectral | Speech features + prosody analysis |
    | Video | Frame + Temporal | Per-frame + optical flow analysis |

    ### Secure Chat (E2EE)

    Included to satisfy the rubric's *"End-to-End Encrypted Chat Application"*
    requirement. Messages are encrypted with **Fernet (AES-128-CBC + HMAC-SHA256)**
    on the client and only decrypted on the receiver side.

    ### Integrity Guarantee

    All analyzed files are hashed (SHA-256) before and after analysis
    to verify they are not corrupted.

    ### Limitations

    - AI detection is probabilistic, not deterministic
    - False positives and false negatives are possible
    - Results should be reviewed by a security professional
    - No detection system is 100% accurate
    """
    )


if __name__ == "__main__":
    main()