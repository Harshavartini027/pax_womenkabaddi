import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from PIL import Image
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Women's Kabaddi Team Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# IMAGE PATHS CONFIGURATION
# ============================================================

# Individual player pie charts - Dictionary mapping player names to image paths
PLAYER_PIE_CHARTS = {
    "ABINA JANET BERIN": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\TP_indiv_Abina.png",
    "ARIVUMATHI": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\Arivimathi.png",
    "GAYATHIRI": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\GAYATHRI.png",
    "KARPAKAVALLI": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\KARPAGAVALLI.png",
    "KARTHIKA": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\TP_indiv_Karthika.png",
    "NARTHIKA": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\NARTHIKA.png",
    "NAVYA": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\tournmnt perf_Navya_Indi.png",
    "SRI HARINI": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\SRI HARINI.png",
    "SUMATHI": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\TP_indiv_Sumathi.png",
    "MAHAL": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\TP_indiv_MahaL.png",
    "HARINI T": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\TP_indiv_Harini T.png"
}

# Other visualization images
AVG_POINTS_CHART = r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\ind_avg points.png"
PERF_VS_ATTENDANCE_CHART = r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\perf vs attend.png"
PLAYER_DETAILS_CHART = r"C:\Users\samyu\Desktop\player_details_analysis.png"

# Top 5 Players Web Charts - Individual radar/web charts
TOP5_WEB_CHARTS = {
    "NARTHIKA": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\narthika_attend_web.png",
    "ARIVUMATHI": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\Arivu_attend_web ch.png",
    "KARPAKAVALLI": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\karpaga_attend_web.png",
    "SUMATHI": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\sumathi_attend_web.png",
    "NAVYA": r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\navya_attend_web.png"
}

RADAR_ALL_CHART = r"C:\Users\samyu\Desktop\radar_all_players.png"

# ============================================================

# Helper function to display images
def display_image(image_path, caption="", use_container_width=True):
    """Display image from path with error handling"""
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            st.image(image, caption=caption, use_container_width=use_container_width)
            return True
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return False
    return False  # Silently return False if image doesn't exist

# ---------------- READ DATA ----------------
@st.cache_data
def load_match_data(match_file):
    match_sheets = pd.read_excel(match_file, sheet_name=None)
    cleaned_sheets = []
    
    for name, df in match_sheets.items():
        # SKIP sheets that are summary/total sheets
        if any(keyword in str(name).upper() for keyword in ["TOTAL", "SUMMARY", "MATCHES"]):
            continue
            
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        df["Tournament"] = str(name)
        cleaned_sheets.append(df)
    
    matches = pd.concat(cleaned_sheets, ignore_index=True)
    matches.columns = matches.columns.astype(str)
    
    # Detect player name column
    possible_name_cols = [c for c in matches.columns if "name" in c.lower() or "player" in c.lower()]
    if possible_name_cols:
        matches.rename(columns={possible_name_cols[0]: "Player"}, inplace=True)
    else:
        st.error("❌ Could not find a column with player names!")
        st.stop()
    
    # Convert numeric data
    for col in matches.columns:
        if col not in ["Player", "Tournament"]:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")
    
    # Compute total points PER ROW (per tournament appearance)
    # Exclude certain columns that shouldn't be summed
    exclude_cols = ["Player", "Tournament", "Present", "Total", "Attendance_pct"]
    numeric_cols = [col for col in matches.select_dtypes("number").columns 
                   if col not in exclude_cols]
    
    matches["Total Points"] = matches[numeric_cols].sum(axis=1)
    
    return matches, cleaned_sheets

@st.cache_data
def load_total_points_data(match_file):
    """Load the TOTAL SCORES BY INDIVIDUAL PLAYER sheet"""
    try:
        df_total = pd.read_excel(match_file, sheet_name="TOTAL SOCRES BY INDIVUAL PLAYE")
        
        # Clean column names (remove trailing spaces)
        df_total.columns = df_total.columns.str.strip()
        
        # Clean player names
        df_total['PLAYERS'] = df_total['PLAYERS'].str.strip().str.upper()
        
        # Rename columns for consistency
        df_total.rename(columns={'PLAYERS': 'Player', 'TOTAL POINTS': 'Total_Points'}, inplace=True)
        
        return df_total
    except Exception as e:
        st.error(f"Error loading total points data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_attendance_data(attendance_file):
    try:
        att_xls = pd.ExcelFile(attendance_file)
        att_frames = []
        
        for sh in att_xls.sheet_names:
            df_sh = pd.read_excel(attendance_file, sheet_name=sh)
            df_sh = df_sh.dropna(how="all").dropna(axis=1, how="all")
            
            # Clean column names
            df_sh.columns = df_sh.columns.str.strip()
            
            # Detect columns
            name_col = None
            present_col = None
            total_col = None
            
            for c in df_sh.columns:
                c_upper = str(c).upper()
                if 'NAME' in c_upper:
                    name_col = c
                elif 'PRESENT' in c_upper:
                    present_col = c
                elif 'TOTAL' in c_upper and 'PRACTICE' in c_upper:
                    total_col = c
            
            if name_col and present_col and total_col:
                df_clean = df_sh[[name_col, present_col, total_col]].copy()
                df_clean.columns = ['Player', 'Present', 'Total']
                df_clean['Month'] = sh
                att_frames.append(df_clean)
        
        if att_frames:
            att_all = pd.concat(att_frames, ignore_index=True)
        else:
            return pd.DataFrame()
        
        # Clean player names
        att_all['Player'] = att_all['Player'].astype(str).str.strip().str.upper()
        
        # Remove NaN and invalid entries
        att_all = att_all[att_all['Player'].notna()]
        att_all = att_all[~att_all['Player'].isin(['NAN', 'NONE', ''])]
        
        # Convert to numeric
        att_all['Present'] = pd.to_numeric(att_all['Present'], errors='coerce')
        att_all['Total'] = pd.to_numeric(att_all['Total'], errors='coerce')
        
        # Remove rows with invalid data
        att_all = att_all.dropna(subset=['Present', 'Total'])
        att_all = att_all[att_all['Total'] > 0]
        
        # Calculate attendance percentage per month
        att_all['Attendance_pct'] = (att_all['Present'] / att_all['Total'] * 100).round(2)
        
        # Aggregate by player (sum present and total, then recalculate percentage)
        att_summary = att_all.groupby('Player').agg({
            'Present': 'sum',
            'Total': 'sum'
        }).reset_index()
        
        att_summary['Attendance_pct'] = (att_summary['Present'] / att_summary['Total'] * 100).round(2)
        
        return att_summary
    
    except Exception as e:
        st.error(f"Error reading attendance file: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# ============================================================
# FILE PATHS - UPDATE THESE TO YOUR LOCAL FILE LOCATIONS
# ============================================================

# IMPORTANT: Update these paths to match your file locations
match_file = r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\Kabaddi match points final.xlsx"
attendance_file = r"C:\Users\samyu\Desktop\PAX-Kabbaddi analysis\Copy_of_PAX_Attendance__processed.xlsx"

# Load data
try:
    matches, cleaned_sheets = load_match_data(match_file)
    total_points_df = load_total_points_data(match_file)  # NEW: Load total points sheet
    attendance = load_attendance_data(attendance_file)
    
    # Normalize player names and REMOVE NaN players
    matches["Player"] = matches["Player"].astype(str).str.strip().str.upper()
    matches = matches[matches["Player"].notna() & (matches["Player"] != "NAN") & (matches["Player"] != "")]
    
    # Store Total Points before merging (to avoid contamination from attendance columns)
    points_only = matches["Total Points"].copy()
    
    # Merge attendance data
    if not attendance.empty:
        matches = matches.merge(attendance, on="Player", how="left")
        # Restore the correct Total Points (not including attendance numbers)
        matches["Total Points"] = points_only
    
    # Merge total points data from the summary sheet
    if not total_points_df.empty:
        matches = matches.merge(total_points_df[['Player', 'Total_Points']], on="Player", how="left")
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏐 Kabaddi Analytics")
st.sidebar.markdown("---")

# Navigation without Overview tab
page = st.sidebar.radio("📊 Navigation", ["👤 Individual Analysis", "🏆 Team Analysis", "📈 Attendance Insights"])

# =====================================================
# 👤 INDIVIDUAL ANALYSIS
# =====================================================
if page == "👤 Individual Analysis":
    st.markdown('<div class="main-header">👤 Individual Player Analysis</div>', unsafe_allow_html=True)
    
    # Get valid players only (no NaN)
    valid_players = sorted([p for p in matches["Player"].unique() if p and str(p).upper() != "NAN"])
    
    if not valid_players:
        st.error("No valid player data available.")
        st.stop()
    
    selected_player = st.selectbox("Select a Player", valid_players, key="individual_player")
    
    # Filter data for ONLY the selected player
    player_df = matches[matches["Player"] == selected_player].copy()
    
    # Get TOTAL POINTS from the summary sheet (Total_Points column)
    if 'Total_Points' in player_df.columns and not player_df['Total_Points'].isna().all():
        total_points_from_sheet = player_df['Total_Points'].iloc[0]
    else:
        # Fallback to calculated total if summary sheet data not available
        total_points_from_sheet = player_df["Total Points"].sum()
    
    # Calculate metrics for THIS PLAYER ONLY
    # Count only valid tournaments (excluding TOTAL/MATCHES entries)
    valid_tournaments = [t for t in player_df["Tournament"].unique() 
                        if not any(x in str(t).upper() for x in ["TOTAL", "MATCHES"])]
    total_matches = len(valid_tournaments)
    
    # Average points per tournament
    avg_points = total_points_from_sheet / total_matches if total_matches > 0 else 0
    
    # Show breakdown by tournament for verification
    with st.expander("View Points Breakdown by Tournament"):
        tournament_breakdown = player_df.groupby("Tournament")["Total Points"].sum().reset_index()
        tournament_breakdown.columns = ["Tournament", "Points in Tournament"]
        st.dataframe(tournament_breakdown, use_container_width=True)
        calculated_total = tournament_breakdown['Points in Tournament'].sum()
        st.info(f"**Total Points from Summary Sheet:** {int(total_points_from_sheet)}")
        st.info(f"**Calculated Total (verification):** {int(calculated_total)}")
        st.info(f"**Average Points Calculation:** {int(total_points_from_sheet)} ÷ {total_matches} = {avg_points:.2f}")
    
    # Player metrics - Showing Total Points from Sheet + Average
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Total Points", int(total_points_from_sheet))
    with col2:
        st.metric("🏆 Tournaments Played", int(total_matches))
    with col3:
        st.metric("📊 Avg Points/Tournament", f"{avg_points:.2f}")
    
    st.markdown("---")
    
    # Performance analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 {selected_player}'s Performance Across Tournaments")
        
        tournament_perf = player_df.groupby("Tournament")["Total Points"].sum().sort_values(ascending=False)
        tournament_perf = tournament_perf[~tournament_perf.index.str.contains("TOTAL|MATCHES", case=False, na=False)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(tournament_perf)))
        bars = ax.bar(range(len(tournament_perf)), tournament_perf.values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(tournament_perf)))
        ax.set_xticklabels([t[:25] + '...' if len(t) > 25 else t for t in tournament_perf.index], 
                          rotation=45, ha='right')
        ax.set_ylabel("Points Scored", fontsize=12, fontweight='bold')
        ax.set_title(f"{selected_player} - Tournament Performance", fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, tournament_perf.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("🏅 Player Rating")
        
        # Calculate based on TOTAL POINTS from summary sheet
        if not total_points_df.empty:
            mean_all = total_points_df['Total_Points'].mean()
            std_all = total_points_df['Total_Points'].std()
        else:
            mean_all = matches.groupby("Player")["Total Points"].sum().mean()
            std_all = matches.groupby("Player")["Total Points"].sum().std()
        
        if total_points_from_sheet >= mean_all + std_all:
            rating = "⭐ EXCELLENT"
            color = "green"
            emoji = "🔥"
        elif total_points_from_sheet >= mean_all:
            rating = "✅ GOOD"
            color = "blue"
            emoji = "👍"
        else:
            rating = "⚪ MODERATE"
            color = "orange"
            emoji = "📊"
        
        # Calculate rank based on total points
        if not total_points_df.empty:
            total_points_df_sorted = total_points_df.sort_values('Total_Points', ascending=False).reset_index(drop=True)
            player_rank = total_points_df_sorted[total_points_df_sorted['Player'] == selected_player].index[0] + 1
        else:
            player_rank = matches.groupby('Player')['Total Points'].sum().rank(ascending=False)[selected_player]
        
        st.markdown(f"""
        <div style='padding: 2rem; background-color: {color}20; border-radius: 10px; border-left: 5px solid {color};'>
            <h2 style='color: {color}; text-align: center;'>{emoji} {rating}</h2>
            <p style='text-align: center; font-size: 1.2rem;'>
                <strong>Total Points:</strong> {int(total_points_from_sheet)}<br>
                <strong>Avg Points/Tournament:</strong> {avg_points:.2f}<br>
                <strong>Rank:</strong> {int(player_rank)} of {len(valid_players)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Round-wise performance - Display image from path
        st.subheader("🎯 Round-wise Breakdown")
        
        # Get pie chart path for selected player
        pie_chart_path = PLAYER_PIE_CHARTS.get(selected_player)
        
        if pie_chart_path and os.path.exists(pie_chart_path):
            display_image(pie_chart_path, caption=f"{selected_player} - Tournament Performance Distribution")
        else:
            st.info(f"ℹ️ Pie chart not available for {selected_player}.\n\n"
                   f"To add a pie chart for this player:\n"
                   f"1. Create a pie chart showing round-wise performance\n"
                   f"2. Save it to your preferred location\n"
                   f"3. Update the PLAYER_PIE_CHARTS dictionary (around line 42) with:\n\n"
                   f'```python\n"{selected_player}": r"C:\\path\\to\\your\\image.png"\n```')
    
    # Add "Average Points per Tournament by Each Player" visualization
    st.markdown("---")
    st.subheader("📊 Average Points per Tournament by Each Player")
    
    if os.path.exists(AVG_POINTS_CHART):
        display_image(AVG_POINTS_CHART, caption="Average Points per Tournament by Each Player")
    else:
        st.info("ℹ️ Average points chart not available. Update AVG_POINTS_CHART path in the code (line 57) to display this visualization.")

# =====================================================
# 🏆 TEAM ANALYSIS
# =====================================================
elif page == "🏆 Team Analysis":
    st.markdown('<div class="main-header">🏆 Team Analysis</div>', unsafe_allow_html=True)
    
    # Added "Comparison with Top 10 Players" - Using Total Points from Summary Sheet
    st.subheader("📊 Comparison - Top 10 Players")
    
    if not total_points_df.empty:
        top_10_df = total_points_df.nlargest(10, 'Total_Points')
        top_10 = pd.Series(top_10_df['Total_Points'].values, index=top_10_df['Player'].values)
    else:
        top_10 = matches.groupby("Player")["Total Points"].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#4ECDC4' for _ in top_10.index]
    bars = ax.barh(range(len(top_10)), top_10.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10.index)
    ax.set_xlabel("Total Points", fontsize=12, fontweight='bold')
    ax.set_title("Top 10 Players Comparison", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, top_10.values)):
        ax.text(val + 1, i, f'{int(val)}', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Team progression
    st.subheader("📈 Team Progression Through Tournament Rounds")
    
    round_columns = [col for col in matches.columns
                     if any(key in col.upper() for key in ["ROUND", "Q-FINAL", "QF", "SEMI", "FINAL"])]
    
    if round_columns:
        round_points = matches[round_columns].sum(numeric_only=True)
        stage_totals = round_points.reset_index()
        stage_totals.columns = ["Round / Stage", "Total Points"]
        stage_totals["Round / Stage"] = stage_totals["Round / Stage"].astype(str).str.upper().str.strip()
        
        def normalize_stage(s):
            s = s.upper()
            if "Q-FINAL" in s or s == "QF" or "QUARTER" in s:
                return "Q-FINAL"
            if "ROUND 1" in s or s == "R1":
                return "ROUND 1"
            if "ROUND 2" in s or s == "R2":
                return "ROUND 2"
            if "ROUND 3" in s or s == "R3":
                return "ROUND 3"
            if "SEMI" in s:
                return "SEMI-FINAL"
            if "FINAL" in s and "SEMI" not in s:
                return "FINAL"
            return s
        
        stage_totals["Round / Stage"] = stage_totals["Round / Stage"].apply(normalize_stage)
        stage_totals = stage_totals.groupby("Round / Stage", as_index=False)["Total Points"].sum()
        
        desired_order = ['ROUND 1', 'ROUND 2', 'ROUND 3', 'Q-FINAL', 'SEMI-FINAL', 'FINAL']
        stage_totals["Round / Stage"] = pd.Categorical(stage_totals["Round / Stage"],
                                                       categories=desired_order, ordered=True)
        stage_totals = stage_totals.sort_values("Round / Stage").dropna(subset=["Round / Stage"])
        
        nonzero = stage_totals[stage_totals["Total Points"] > 0]
        if not nonzero.empty:
            last_stage = nonzero["Round / Stage"].iloc[-1]
            st.success(f"🏆 **Highest Stage Reached:** {last_stage}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(stage_totals)))
            bars = ax.bar(stage_totals["Round / Stage"].astype(str), stage_totals["Total Points"], 
                         color=colors, edgecolor='black', linewidth=2)
            ax.set_xlabel("Tournament Round", fontsize=12, fontweight='bold')
            ax.set_ylabel("Total Points", fontsize=12, fontweight='bold')
            ax.set_title("Team Performance Across Tournament Stages", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            
            for bar, val in zip(bars, stage_totals["Total Points"]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📊 Stage Statistics")
            for idx, row in stage_totals.iterrows():
                st.metric(row["Round / Stage"], f"{int(row['Total Points'])} points")
    
    # Quarter Final Analysis
    st.markdown("---")
    st.subheader("🏅 Quarter Final Performance Analysis")
    
    qfinal_candidates = [col for col in matches.columns 
                        if any(x in col.upper() for x in ["Q-FINAL", "QF", "QUARTER"])]
    
    if qfinal_candidates:
        qcol = qfinal_candidates[0]
        qf_summary = matches.groupby("Player")[qcol].sum().reset_index().sort_values(by=qcol, ascending=False)
        qf_summary = qf_summary[qf_summary[qcol] > 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(qf_summary.head(10).style.background_gradient(subset=[qcol], cmap='Oranges'), 
                        use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_qf = qf_summary.head(10)
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_qf)))
            bars = ax.barh(range(len(top_qf)), top_qf[qcol].values, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_yticks(range(len(top_qf)))
            ax.set_yticklabels(top_qf["Player"])
            ax.set_xlabel("Points", fontsize=12, fontweight='bold')
            ax.set_title("Top 10 Players - Quarter Final", fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            for i, (bar, val) in enumerate(zip(bars, top_qf[qcol].values)):
                ax.text(val + 0.3, i, f'{int(val)}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    

# =====================================================
# 📈 ATTENDANCE INSIGHTS
# =====================================================
else:
    st.markdown('<div class="main-header">📈 Attendance & Performance Insights</div>', unsafe_allow_html=True)
    
    if attendance.empty:
        st.warning("⚠️ Attendance data not available. Please check the attendance file.")
        st.info("Expected file: Copy_of_PAX_Attendance__processed.xlsx")
    else:
        # Attendance overview
        st.subheader("📊 Attendance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        avg_att = attendance["Attendance_pct"].mean()
        max_att = attendance["Attendance_pct"].max()
        min_att = attendance["Attendance_pct"].min()
        total_players_att = len(attendance)
        
        col1.metric("📈 Average Attendance", f"{avg_att:.1f}%")
        col2.metric("🌟 Highest Attendance", f"{max_att:.1f}%")
        col3.metric("⚠️ Lowest Attendance", f"{min_att:.1f}%")
        col4.metric("👥 Players Tracked", total_players_att)
        
        st.markdown("---")
        
        # Attendance Categories with proper color coding
        st.subheader("🎯 Attendance Categories")
        
        # Categorize attendance with proper color coding
        def categorize_attendance(pct):
            if pct >= 90:
                return "Excellent (≥90%)"
            elif pct >= 75:
                return "Good (75-89%)"
            elif pct >= 60:
                return "Average (60-74%)"
            else:
                return "Poor (<60%)"
        
        attendance["Category"] = attendance["Attendance_pct"].apply(categorize_attendance)
        cat_counts = attendance["Category"].value_counts()
        
        # Updated colors - Green for good, Red for bad
        color_map = {
            "Excellent (≥90%)": '#2ecc71',
            "Good (75-89%)": '#95e1d3',
            "Average (60-74%)": '#f39c12',
            "Poor (<60%)": '#e74c3c'
        }
        
        colors_cat = [color_map.get(cat, '#95a5a6') for cat in cat_counts.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, texts, autotexts = ax.pie(cat_counts.values, labels=cat_counts.index, 
                                           autopct='%1.1f%%', startangle=90,
                                           colors=colors_cat, 
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title("Attendance Category Distribution", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Only Top 10 Attendance
        st.markdown("---")
        st.subheader("🌟 Top 10 Attendance")
        
        top_att = attendance.nlargest(10, "Attendance_pct")[["Player", "Attendance_pct", "Present", "Total"]]
        st.dataframe(top_att.style.background_gradient(subset=["Attendance_pct"], cmap='Greens'),
                    use_container_width=True, hide_index=True)
        
        # Performance vs Attendance
        st.markdown("---")
        st.subheader("🔬 Performance vs Attendance Correlation Analysis")
        
        if os.path.exists(PERF_VS_ATTENDANCE_CHART):
            display_image(PERF_VS_ATTENDANCE_CHART, caption="Performance vs Attendance: Comprehensive Analysis")
        else:
            st.info("ℹ️ Performance vs Attendance chart not available. Update PERF_VS_ATTENDANCE_CHART path in the code (line 58) to display this visualization.")
        
        # Player-wise detailed analysis
        st.markdown("---")
        st.subheader("🔍 Player-wise Detailed Analysis")
        
        if os.path.exists(PLAYER_DETAILS_CHART):
            display_image(PLAYER_DETAILS_CHART, caption="Player-wise Detailed Analysis")
        else:
            # Generate the table if image doesn't exist - Use Total_Points from summary sheet
            if not total_points_df.empty and not attendance.empty:
                # Merge total points with attendance
                perf_att = total_points_df.merge(attendance, on="Player", how="left")
                
                # Calculate tournaments played
                tournaments_played = matches.groupby("Player")["Tournament"].nunique().reset_index()
                tournaments_played.columns = ["Player", "Tournaments"]
                
                perf_att = perf_att.merge(tournaments_played, on="Player", how="left")
                perf_att["Avg Points per Tournament"] = perf_att["Total_Points"] / perf_att["Tournaments"]
                perf_att = perf_att.dropna(subset=["Attendance_pct"])
                
                if not perf_att.empty:
                    perf_att = perf_att.sort_values("Total_Points", ascending=False)
                    
                    # Display as a properly formatted table
                    st.dataframe(
                        perf_att[["Player", "Total_Points", "Tournaments", "Avg Points per Tournament", "Attendance_pct"]]
                        .style.background_gradient(subset=["Avg Points per Tournament"], cmap='Greens')
                        .background_gradient(subset=["Attendance_pct"], cmap='Blues')
                        .format({
                            "Total_Points": "{:.0f}",
                            "Tournaments": "{:.0f}",
                            "Avg Points per Tournament": "{:.2f}",
                            "Attendance_pct": "{:.2f}%"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
        
        # Radar Chart Section - Top 5 Players Web Charts
        st.markdown("---")
        st.subheader("🎯 Top 5 Players - Performance Web Charts")
        
        # Use the TOP5_WEB_CHARTS dictionary defined at the top
        # Check if any images exist
        existing_charts = {name: path for name, path in TOP5_WEB_CHARTS.items() if os.path.exists(path)}
        
        if existing_charts:
            # Display all existing web charts in a grid layout (2 columns)
            col1, col2 = st.columns(2)
            
            # Display all charts
            for idx, (player_name, chart_path) in enumerate(existing_charts.items()):
                with col1 if idx % 2 == 0 else col2:
                    display_image(chart_path, caption=f"{player_name} - Performance Web Chart")
        else:
            st.info("📊 Performance web charts not configured. To add web charts:\n\n"
                   "1. Create radar/web charts for your top players\n"
                   "2. Save them to your desktop or preferred location\n"
                   "3. Update the TOP5_WEB_CHARTS dictionary in the code (around line 60) with the correct paths\n\n"
                   "Example:\n```python\nTOP5_WEB_CHARTS = {\n"
                   "    'KARPAKAVALLI': r'C:\\Users\\samyu\\Desktop\\karpakavalli_web.png',\n"
                   "    'SRI HARINI': r'C:\\Users\\samyu\\Desktop\\sriharini_web.png',\n"
                   "}\n```")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;'>
        <h3 style='color: #FF6B35;'>🏐 Kabaddi Analytics Dashboard</h3>
        <p style='color: #6c757d;'>Developed for College Sports Analytics | Data-Driven Team Performance Insights</p>
    </div>
""", unsafe_allow_html=True)
