import streamlit as st
import pandas as pd
import json
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentiment_analyzer import analyze_sentiment_and_tone
from consistency_checker import analyze_review_consistency
from pairwise_comparator import compare_reviews_pairwise
from bias_detector import detect_bias


# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Helper functions for handling list data
def format_list_data(data, field_name, default="Not analyzed"):
    """Format list data for display in metrics"""
    field_data = data.get(field_name, default)
    
    if isinstance(field_data, list) and field_data:
        if len(field_data) == 1:
            return str(field_data[0])
        else:
            # Show most common item for lists with multiple items
            if all(isinstance(item, str) for item in field_data):
                counter = Counter(field_data)
                most_common = counter.most_common(1)[0]
                return f"{most_common[0]} ({most_common[1]}/{len(field_data)})"
            else:
                return f"{len(field_data)} items"
    elif isinstance(field_data, bool):
        return "✅ Yes" if field_data else "❌ No"
    else:
        return str(field_data) if field_data is not None else default

# def format_bias_type_display(bias_type_data):
#     """Format bias type for better display - no truncation"""
#     if isinstance(bias_type_data, list):
#         if len(bias_type_data) == 1:
#             return str(bias_type_data[0]) if bias_type_data[0] else "None"
#         else:
#             # Join multiple bias types with line breaks for better display
#             valid_types = [str(bt) for bt in bias_type_data if bt and str(bt).lower() != 'none']
#             return valid_types if valid_types else ["None"]
#     else:
#         return str(bias_type_data) if bias_type_data and str(bias_type_data).lower() != 'none' else "None"

def format_bias_type_display(bias_type_data):
    """Format bias type for better display - no truncation, exclude None"""
    if isinstance(bias_type_data, list):
        if len(bias_type_data) == 1:
            bias_value = str(bias_type_data[0]) if bias_type_data[0] else "None"
            return bias_value if bias_value.lower() != 'none' else "None"
        else:
            # Join multiple bias types, excluding "None" values
            valid_types = []
            for bt in bias_type_data:
                if isinstance(bt, list):
                    valid_subtypes = [str(subbt) for subbt in bt if subbt and str(subbt).lower() != 'none']
                    valid_types.extend(valid_subtypes)
                elif bt and str(bt).lower() != 'none':
                    valid_types.append(str(bt))
            return valid_types if valid_types else ["None"]
    else:
        bias_value = str(bias_type_data) if bias_type_data else "None"
        return bias_value if bias_value.lower() != 'none' else "None"

def create_distribution_chart(data, field_name, title, chart_type="bar"):
    """Create professional distribution charts using Plotly"""
    field_data = data.get(field_name, [])
    
    if isinstance(field_data, list) and field_data and len(field_data) > 1:
        # Handle nested lists by flattening them
        flattened_data = []
        for item in field_data:
            if isinstance(item, list):
                flattened_data.extend(item)
            elif item is not None:
                flattened_data.append(item)
        
        if flattened_data:
            try:
                counter = Counter(flattened_data)
                
                # Create DataFrame for plotting
                chart_df = pd.DataFrame.from_dict(counter, orient='index', columns=['Count']).reset_index()
                chart_df.columns = [field_name.title(), 'Count']
                
                if chart_type == "pie":
                    fig = px.pie(chart_df, values='Count', names=field_name.title(), 
                               title=f"{title} Distribution",
                               color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                else:
                    fig = px.bar(chart_df, x=field_name.title(), y='Count', 
                               title=f"{title} Distribution",
                               color='Count',
                               color_continuous_scale='viridis')
                    
                    # Add value labels on bars
                    fig.update_traces(texttemplate='%{y}', textposition='outside')
                
                fig.update_layout(
                    showlegend=True if chart_type == "pie" else False,
                    height=400,
                    font=dict(size=12),
                    title_font_size=16
                )
                
                return fig, chart_df
                
            except TypeError:
                return None, None
    
    return None, None

def create_confidence_chart(confidence_scores):
    """Create confidence score visualization"""
    if isinstance(confidence_scores, list) and len(confidence_scores) > 1:
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=list(range(1, len(confidence_scores) + 1)),
            y=confidence_scores,
            mode='markers+lines',
            name='Confidence Score',
            marker=dict(size=10, color='blue'),
            line=dict(width=2)
        ))
        
        # Add average line
        avg_score = sum(confidence_scores) / len(confidence_scores)
        fig.add_hline(y=avg_score, line_dash="dash", line_color="red",
                     annotation_text=f"Average: {avg_score:.2f}")
        
        fig.update_layout(
            title="Confidence Scores Across Reviews",
            xaxis_title="Review Number",
            yaxis_title="Confidence Score",
            height=400,
            showlegend=True
        )
        
        return fig
    
    return None

def display_list_details(data, field_name, title):
    """Display detailed information for list fields"""
    field_data = data.get(field_name, [])
    
    if isinstance(field_data, list) and field_data:
        st.write(f"**{title}:**")
        for i, item in enumerate(field_data):
            st.write(f"**Review {i+1}:** {item}")
    elif field_data and not isinstance(field_data, list):
        st.write(f"**{title}:** {field_data}")

def display_individual_bias_results(data):
    """Display bias detection results for each review individually with improved formatting"""
    bias_detected = data.get('bias_detected', [])
    bias_type = data.get('bias_type', [])
    confidence_score = data.get('confidence_score', [])
    evidence = data.get('evidence', [])
    suggestions = data.get('suggestion_for_improvements', [])
    
    # Determine number of reviews
    if isinstance(bias_detected, list):
        num_reviews = len(bias_detected)
    elif isinstance(bias_type, list):
        num_reviews = len(bias_type)
    elif isinstance(confidence_score, list):
        num_reviews = len(confidence_score)
    else:
        num_reviews = 1
    
    # If we have multiple reviews, display each one
    if num_reviews > 1:
        st.subheader("📋 Individual Review Results")
        
        for i in range(num_reviews):
            st.write(f"### Review {i+1}")
            
            # Create columns for each review
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Bias detected status
                review_bias = bias_detected[i] if isinstance(bias_detected, list) and i < len(bias_detected) else bias_detected
                if review_bias:
                    st.error("🚨 BIAS DETECTED")
                else:
                    st.success("✅ NO BIAS DETECTED")
                
                # Confidence score
                review_confidence = confidence_score[i] if isinstance(confidence_score, list) and i < len(confidence_score) else confidence_score
                if isinstance(review_confidence, (int, float)):
                    st.metric("Confidence", f"{review_confidence:.2f}")
                else:
                    st.metric("Confidence", str(review_confidence))
            
            with col2:
                # Bias type - Improved display without truncation
                review_bias_type = bias_type[i] if isinstance(bias_type, list) and i < len(bias_type) else bias_type
                formatted_bias_type = format_bias_type_display(review_bias_type)
                
                st.write("**Bias Type(s):**")
                if isinstance(formatted_bias_type, list):
                    for bt in formatted_bias_type:
                        if bt != "None":
                            st.warning(f"⚠️ {bt}")
                        else:
                            st.info("✅ No bias detected")
                else:
                    if formatted_bias_type != "None":
                        st.warning(f"⚠️ {formatted_bias_type}")
                    else:
                        st.info("✅ No bias detected")
            
            # Evidence for this review
            review_evidence = evidence[i] if isinstance(evidence, list) and i < len(evidence) else evidence
            if review_evidence:
                st.write("**Evidence:**")
                with st.expander("View Evidence Details", expanded=False):
                    if isinstance(review_evidence, list):
                        for item in review_evidence:
                            st.warning(f"• {item}")
                    else:
                        st.warning(review_evidence)
            
            # Suggestions for this review
            review_suggestions = suggestions[i] if isinstance(suggestions, list) and i < len(suggestions) else suggestions
            if review_suggestions:
                st.write("**Suggestions for Improvement:**")
                with st.expander("View Improvement Suggestions", expanded=False):
                    if isinstance(review_suggestions, list):
                        for item in review_suggestions:
                            st.info(f"• {item}")
                    else:
                        st.info(review_suggestions)
            
            # Add separator between reviews
            if i < num_reviews - 1:
                st.divider()
    
    else:
        # Single review case
        st.subheader("📋 Review Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if bias_detected:
                st.error("🚨 BIAS DETECTED")
            else:
                st.success("✅ NO BIAS DETECTED")
            
            if isinstance(confidence_score, (int, float)):
                st.metric("Confidence", f"{confidence_score:.2f}")
            else:
                st.metric("Confidence", str(confidence_score))
        
        with col2:
            formatted_bias_type = format_bias_type_display(bias_type)
            st.write("**Bias Type(s):**")
            if isinstance(formatted_bias_type, list):
                for bt in formatted_bias_type:
                    if bt != "None":
                        st.warning(f"⚠️ {bt}")
                    else:
                        st.info("✅ No bias detected")
            else:
                if formatted_bias_type != "None":
                    st.warning(f"⚠️ {formatted_bias_type}")
                else:
                    st.info("✅ No bias detected")
        
        # Evidence
        if evidence:
            st.write("**Evidence:**")
            with st.expander("View Evidence Details", expanded=False):
                if isinstance(evidence, list):
                    for item in evidence:
                        st.warning(f"• {item}")
                else:
                    st.warning(evidence)
        
        # Suggestions
        if suggestions:
            st.write("**Suggestions for Improvement:**")
            with st.expander("View Improvement Suggestions", expanded=False):
                if isinstance(suggestions, list):
                    for item in suggestions:
                        st.info(f"• {item}")
                else:
                    st.info(suggestions)

# Add custom CSS to improve tab display and general styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        overflow-x: auto;
        white-space: nowrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        padding-left: 12px;
        padding-right: 12px;
        font-size: 14px;
        min-width: auto;
        flex-shrink: 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff4b4b;
    }
    
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 12px;
            padding-left: 8px;
            padding-right: 8px;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Review Bias Detection & Analysis System")
st.markdown("*Professional academic review analysis with comprehensive bias detection*")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 File Upload")
    file = st.file_uploader(
        label="Upload files (CSV, JSON):", 
        accept_multiple_files=False, 
        type=['csv', 'json']
    )
    
    # Process uploaded file
    if file:
        try:
            if file.name.split('.')[-1] == 'csv':
                file_type = 'csv'
                st.session_state.data = pd.read_csv(file)
            elif file.name.split('.')[-1] == 'json':
                file_type = 'json'
                st.session_state.data = json.load(file)
            st.success(f"✅ {file.name} uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error uploading file: {str(e)}")
    
    # Analysis controls
    if st.session_state.data is not None:
        st.header("🔍 Analysis Controls")
        if st.button('🚀 Start Complete Analysis', type="primary"):
            st.session_state.analysis_complete = False
            with st.spinner("Running comprehensive analysis..."):
                try:
                    data = st.session_state.data
                    
                    # Sentiment Analysis
                    sentiment, sentiment_reason, tone, tone_reason = analyze_sentiment_and_tone(data[0]['review_contents'])
                    data[0]['sentiment'] = sentiment
                    data[0]['sentiment_reason'] = sentiment_reason
                    data[0]['tone'] = tone
                    data[0]['tone_reason'] = tone_reason
                    
                    # Consistency Analysis
                    consistency, consistency_reason = analyze_review_consistency(data[0]['review_contents'])
                    data[0]['consistency'] = consistency
                    data[0]['consistency_reason'] = consistency_reason
                    
                    # Pairwise Comparison
                    is_consistent_with_others, alignment_score, contradictory_points, possible_bias_flags, summary_of_differences = compare_reviews_pairwise(input_data=data)
                    data[0]['is_consistent_with_others'] = is_consistent_with_others
                    data[0]['alignment_score'] = alignment_score
                    data[0]['contradictory_points'] = contradictory_points
                    data[0]['possible_bias_flags'] = possible_bias_flags
                    data[0]['summary_of_differences'] = summary_of_differences
                    
                    # Bias Detection
                    bias_detected, bias_type, confidence_score, evidence, suggestion_for_improvements = detect_bias(input_data=data)
                    data[0]['bias_detected'] = bias_detected
                    data[0]['bias_type'] = bias_type
                    data[0]['confidence_score'] = confidence_score
                    data[0]['evidence'] = evidence
                    data[0]['suggestion_for_improvements'] = suggestion_for_improvements
                    
                    st.session_state.data = data
                    st.session_state.analysis_complete = True
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

# Main content area with tabs
if st.session_state.data is not None:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data", 
        "😊 Sentiment", 
        "🔄 Consistency", 
        "⚖️ Comparison", 
        "🚨 Bias", 
        "💾 Download"
    ])
    
    with tab1:
        st.header("📊 Input Data Overview")
        data = st.session_state.data
        
        # Basic info with improved styling
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", len(data))
        with col2:
            if isinstance(data[0], dict) and 'review_contents' in data[0]:
                st.metric("Review Items", len(data[0]['review_contents']))
        with col3:
            st.metric("Analysis Status", "✅ Complete" if st.session_state.analysis_complete else "⏳ Pending")
        
        # Display raw data
        with st.expander("🔍 View Raw Data", expanded=False):
            st.json(data[0])
        
        # Review contents preview
        if isinstance(data[0], dict) and 'review_contents' in data[0]:
            st.subheader("📝 Review Contents Preview")
            for i, item in enumerate(data[0]['review_contents']):
                with st.expander(f"Review Item {i+1}", expanded=False):
                    st.write(item)
    
    # with tab2:
    #     st.header("😊 Sentiment & Tone Analysis")
        
    #     if st.session_state.analysis_complete and 'sentiment' in st.session_state.data[0]:
    #         data = st.session_state.data[0]
            
    #         # Metrics with improved layout
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             sentiment_display = format_list_data(data, 'sentiment')
    #             st.metric("Overall Sentiment", sentiment_display)
                    
    #         with col2:
    #             tone_display = format_list_data(data, 'tone')
    #             st.metric("Overall Tone", tone_display)
            
    #         # Distribution charts with Plotly
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             fig, chart_data = create_distribution_chart(data, 'sentiment', 'Sentiment')
    #             if fig:
    #                 st.plotly_chart(fig, use_container_width=True)
                    
    #         with col2:
    #             fig, chart_data = create_distribution_chart(data, 'tone', 'Tone')
    #             if fig:
    #                 st.plotly_chart(fig, use_container_width=True)
            
    #         # Detailed analysis
    #         st.subheader("📋 Detailed Analysis")
    #         with st.expander("View Sentiment Analysis Details", expanded=False):
    #             display_list_details(data, 'sentiment_reason', 'Sentiment Analysis')
    #         with st.expander("View Tone Analysis Details", expanded=False):
    #             display_list_details(data, 'tone_reason', 'Tone Analysis')
                
    #     else:
    #         st.info("🔄 Run analysis to see sentiment and tone results")

    with tab2:
        st.header("😊 Sentiment & Tone Analysis")
        
        if st.session_state.analysis_complete and 'sentiment' in st.session_state.data[0]:
            data = st.session_state.data[0]
            
            # Overall metrics with improved layout
            col1, col2 = st.columns(2)
            with col1:
                sentiment_display = format_list_data(data, 'sentiment')
                st.metric("Overall Sentiment", sentiment_display)
                    
            with col2:
                tone_display = format_list_data(data, 'tone')
                st.metric("Overall Tone", tone_display)
            
            # Distribution charts with Plotly
            col1, col2 = st.columns(2)
            with col1:
                fig, chart_data = create_distribution_chart(data, 'sentiment', 'Sentiment')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
            with col2:
                fig, chart_data = create_distribution_chart(data, 'tone', 'Tone')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Individual Review Results
            sentiment_data = data.get('sentiment', [])
            tone_data = data.get('tone', [])
            sentiment_reason_data = data.get('sentiment_reason', [])
            tone_reason_data = data.get('tone_reason', [])
            
            # Determine number of reviews
            num_reviews = 1
            if isinstance(sentiment_data, list):
                num_reviews = max(num_reviews, len(sentiment_data))
            if isinstance(tone_data, list):
                num_reviews = max(num_reviews, len(tone_data))
            
            if num_reviews > 1:
                st.subheader("📋 Individual Review Results")
                
                for i in range(num_reviews):
                    with st.expander(f"📝 Review {i+1} - Sentiment & Tone Analysis", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Individual sentiment
                            review_sentiment = sentiment_data[i] if isinstance(sentiment_data, list) and i < len(sentiment_data) else sentiment_data
                            st.write("**Sentiment:**")
                            if str(review_sentiment).lower() in ['positive', 'very positive']:
                                st.success(f"😊 {review_sentiment}")
                            elif str(review_sentiment).lower() in ['negative', 'very negative']:
                                st.error(f"😞 {review_sentiment}")
                            else:
                                st.info(f"😐 {review_sentiment}")
                            
                            # Individual sentiment reason
                            review_sentiment_reason = sentiment_reason_data[i] if isinstance(sentiment_reason_data, list) and i < len(sentiment_reason_data) else sentiment_reason_data
                            if review_sentiment_reason:
                                st.write("**Sentiment Analysis:**")
                                st.write(f"💭 {review_sentiment_reason}")
                        
                        with col2:
                            # Individual tone
                            review_tone = tone_data[i] if isinstance(tone_data, list) and i < len(tone_data) else tone_data
                            st.write("**Tone:**")
                            if str(review_tone).lower() in ['professional', 'formal', 'positive']:
                                st.success(f"🎯 {review_tone}")
                            elif str(review_tone).lower() in ['aggressive', 'harsh', 'negative']:
                                st.error(f"⚠️ {review_tone}")
                            else:
                                st.info(f"📝 {review_tone}")
                            
                            # Individual tone reason
                            review_tone_reason = tone_reason_data[i] if isinstance(tone_reason_data, list) and i < len(tone_reason_data) else tone_reason_data
                            if review_tone_reason:
                                st.write("**Tone Analysis:**")
                                st.write(f"💭 {review_tone_reason}")
            else:
                # Single review detailed analysis
                st.subheader("📋 Detailed Analysis")
                with st.expander("📝 Sentiment & Tone Details", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Sentiment Analysis:**")
                        if sentiment_reason_data:
                            st.info(f"💭 {sentiment_reason_data}")
                    
                    with col2:
                        st.write("**Tone Analysis:**")
                        if tone_reason_data:
                            st.info(f"💭 {tone_reason_data}")
                
        else:
            st.info("🔄 Run analysis to see sentiment and tone results")
    
    # with tab3:
    #     st.header("🔄 Internal Consistency Analysis")
        
    #     if st.session_state.analysis_complete and 'consistency' in st.session_state.data[0]:
    #         data = st.session_state.data[0]
            
    #         # Consistency metric
    #         consistency_display = format_list_data(data, 'consistency')
    #         st.metric("Consistency Status", consistency_display)
            
    #         # Distribution chart
    #         fig, chart_data = create_distribution_chart(data, 'consistency', 'Consistency', "pie")
    #         if fig:
    #             st.plotly_chart(fig, use_container_width=True)
            
    #         # Detailed reasoning
    #         st.subheader("📋 Analysis Details")
    #         with st.expander("View Consistency Analysis Details", expanded=True):
    #             display_list_details(data, 'consistency_reason', 'Consistency Analysis')
                
    #     else:
    #         st.info("🔄 Run analysis to see consistency results")
    with tab3:
        st.header("🔄 Internal Consistency Analysis")
        
        if st.session_state.analysis_complete and 'consistency' in st.session_state.data[0]:
            data = st.session_state.data[0]
            
            # Overall consistency metric
            consistency_display = format_list_data(data, 'consistency')
            st.metric("Overall Consistency Status", consistency_display)
            
            # Distribution chart
            fig, chart_data = create_distribution_chart(data, 'consistency', 'Consistency', "pie")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual Review Results
            consistency_data = data.get('consistency', [])
            consistency_reason_data = data.get('consistency_reason', [])
            
            # Determine number of reviews
            num_reviews = 1
            if isinstance(consistency_data, list):
                num_reviews = len(consistency_data)
            
            if num_reviews > 1:
                st.subheader("📋 Individual Review Consistency Results")
                
                for i in range(num_reviews):
                    with st.expander(f"🔄 Review {i+1} - Consistency Analysis", expanded=False):
                        # Individual consistency status
                        review_consistency = consistency_data[i] if isinstance(consistency_data, list) and i < len(consistency_data) else consistency_data
                        
                        st.write("**Consistency Status:**")
                        if isinstance(review_consistency, bool):
                            if review_consistency:
                                st.success("✅ Consistent")
                            else:
                                st.error("❌ Inconsistent")
                        else:
                            st.info(f"📊 {review_consistency}")
                        
                        # Individual consistency reason
                        review_consistency_reason = consistency_reason_data[i] if isinstance(consistency_reason_data, list) and i < len(consistency_reason_data) else consistency_reason_data
                        if review_consistency_reason:
                            st.write("**Consistency Analysis:**")
                            st.write(f"💭 {review_consistency_reason}")
            else:
                # Single review detailed analysis
                st.subheader("📋 Detailed Consistency Analysis")
                with st.expander("🔄 Consistency Details", expanded=True):
                    if consistency_reason_data:
                        st.info(f"💭 {consistency_reason_data}")
                
        else:
            st.info("🔄 Run analysis to see consistency results")
    
    with tab4:
        st.header("⚖️ Inter-Review Comparison Analysis")
        
        if st.session_state.analysis_complete and 'alignment_score' in st.session_state.data[0]:
            data = st.session_state.data[0]
            
            # Metrics with improved layout
            col1, col2, col3 = st.columns(3)
            with col1:
                alignment_display = format_list_data(data, 'is_consistent_with_others')
                st.metric("Overall Alignment", alignment_display)
            
            with col2:
                score_data = data.get('alignment_score', [])
                if isinstance(score_data, list) and score_data:
                    avg_score = sum(score_data) / len(score_data)
                    st.metric("Avg Alignment Score", f"{avg_score:.2f}")
                else:
                    st.metric("Alignment Score", str(score_data))
            
            with col3:
                contradictions = data.get('contradictory_points', [])
                total_contradictions = len(contradictions) if isinstance(contradictions, list) else 0
                st.metric("Contradictory Points", total_contradictions)
            
            # Alignment score visualization
            if isinstance(score_data, list) and len(score_data) > 1:
                fig = create_confidence_chart(score_data)
                if fig:
                    fig.update_layout(title="Alignment Scores Across Review Pairs", 
                                    yaxis_title="Alignment Score")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis with expandable sections
            st.subheader("📋 Detailed Comparison Results")
            
            # Contradictory points
            contradictions = data.get('contradictory_points', [])
            if contradictions:
                with st.expander("⚠️ Contradictory Points Found", expanded=True):
                    if isinstance(contradictions, list):
                        for i, point in enumerate(contradictions):
                            st.warning(f"**Review Pair {i+1}:** {point}")
                    else:
                        st.warning(contradictions)
            
            # Bias flags
            bias_flags = data.get('possible_bias_flags', [])
            if bias_flags:
                with st.expander("🚩 Possible Bias Flags", expanded=True):
                    if isinstance(bias_flags, list):
                        for i, flag in enumerate(bias_flags):
                            st.error(f"🚩 **Review {i+1}:** {flag}")
                    else:
                        st.error(f"🚩 {bias_flags}")
            
            # Summary of differences
            summary_diff = data.get('summary_of_differences', [])
            if summary_diff:
                with st.expander("📊 Summary of Key Differences", expanded=False):
                    if isinstance(summary_diff, list):
                        for i, diff in enumerate(summary_diff):
                            st.info(f"**Review Pair {i+1}:** {diff}")
                    else:
                        st.info(summary_diff)
                
        else:
            st.info("🔄 Run analysis to see inter-review comparison results")
    
    with tab5:
        st.header("🚨 Comprehensive Bias Detection Results")
        
        if st.session_state.analysis_complete and 'bias_detected' in st.session_state.data[0]:
            data = st.session_state.data[0]
            
            # Overall summary metrics
            bias_detected = data.get('bias_detected', [])
            confidence_scores = data.get('confidence_score', [])
            
            # Calculate summary statistics
            # if isinstance(bias_detected, list):
            #     total_reviews = len(bias_detected)
            #     biased_reviews = sum(1 for b in bias_detected if b) if all(isinstance(b, bool) for b in bias_detected) else len([b for b in bias_detected if b])
            # Calculate summary statistics
            if isinstance(bias_detected, list):
                total_reviews = len(bias_detected)
                if all(isinstance(b, bool) for b in bias_detected):
                    biased_reviews = sum(1 for b in bias_detected if b)
                else:
                    # For bias_type lists, count reviews that have bias types other than "None"
                    bias_type_data = data.get('bias_type', [])
                    if isinstance(bias_type_data, list):
                        biased_reviews = 0
                        for i, bias_types in enumerate(bias_type_data):
                            if isinstance(bias_types, list):
                                # Check if any bias type in the list is not "None"
                                if any(bt and str(bt).lower() != 'none' for bt in bias_types):
                                    biased_reviews += 1
                            elif bias_types and str(bias_types).lower() != 'none':
                                biased_reviews += 1
                    else:
                        biased_reviews = len([b for b in bias_detected if b])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", total_reviews)
                with col2:
                    st.metric("Biased Reviews", biased_reviews)
                with col3:
                    if isinstance(confidence_scores, list) and confidence_scores:
                        avg_confidence = sum(confidence_scores) / len(confidence_scores)
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                # Overall status with improved styling
                if biased_reviews > 0:
                    st.error(f"🚨 **BIAS DETECTED** in {biased_reviews} out of {total_reviews} reviews")
                else:
                    st.success("✅ **NO BIAS DETECTED** in any reviews")
                
                # Confidence score visualization
                if isinstance(confidence_scores, list) and len(confidence_scores) > 1:
                    fig = create_confidence_chart(confidence_scores)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                # Show distribution of bias types
                bias_types = data.get('bias_type', [])
                if isinstance(bias_types, list) and bias_types:
                    st.subheader("📊 Bias Type Distribution")
                    fig, chart_data = create_distribution_chart(data, 'bias_type', 'Bias Types', "pie")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Single review case
                col1, col2 = st.columns(2)
                with col1:
                    if bias_detected:
                        st.error("🚨 BIAS DETECTED")
                    else:
                        st.success("✅ NO BIAS DETECTED")
                with col2:
                    if isinstance(confidence_scores, (int, float)):
                        st.metric("Confidence Score", f"{confidence_scores:.2f}")
            
            # Display individual review results with improved formatting
            display_individual_bias_results(data)
                
        else:
            st.info("🔄 Run analysis to see comprehensive bias detection results")
    
    with tab6:
        st.header("💾 Export & Download Results")
        
        if st.session_state.analysis_complete:
            data = st.session_state.data
            
            # Full analysis download
            download_data = json.dumps(data, indent=4)
            st.download_button(
                label='📥 Download Complete Analysis (JSON)',
                data=download_data,
                file_name='complete_reviews_analysis.json',
                mime='application/json'
            )
            
            # Individual analysis downloads with improved layout
            st.subheader("📋 Individual Component Downloads")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment analysis only
                if 'sentiment' in data[0]:
                    sentiment_data = {
                        'sentiment': data[0].get('sentiment'),
                        'sentiment_reason': data[0].get('sentiment_reason'),
                        'tone': data[0].get('tone'),
                        'tone_reason': data[0].get('tone_reason')
                    }
                    st.download_button(
                        label='😊 Download Sentiment Analysis',
                        data=json.dumps(sentiment_data, indent=4),
                        file_name='sentiment_analysis.json',
                        mime='application/json'
                    )
                
                # Consistency analysis only
                if 'consistency' in data[0]:
                    consistency_data = {
                        'consistency': data[0].get('consistency'),
                        'consistency_reason': data[0].get('consistency_reason')
                    }
                    st.download_button(
                        label='🔄 Download Consistency Analysis',
                        data=json.dumps(consistency_data, indent=4),
                        file_name='consistency_analysis.json',
                        mime='application/json'
                    )
            
            with col2:
                # Inter-review comparison only
                if 'alignment_score' in data[0]:
                    comparison_data = {
                        'is_consistent_with_others': data[0].get('is_consistent_with_others'),
                        'alignment_score': data[0].get('alignment_score'),
                        'contradictory_points': data[0].get('contradictory_points'),
                        'possible_bias_flags': data[0].get('possible_bias_flags'),
                        'summary_of_differences': data[0].get('summary_of_differences')
                    }
                    st.download_button(
                        label='⚖️ Download Inter-Review Comparison',
                        data=json.dumps(comparison_data, indent=4),
                        file_name='inter_review_comparison.json',
                        mime='application/json'
                    )
                
                # Bias detection only
                if 'bias_detected' in data[0]:
                    bias_data = {
                        'bias_detected': data[0].get('bias_detected'),
                        'bias_type': data[0].get('bias_type'),
                        'confidence_score': data[0].get('confidence_score'),
                        'evidence': data[0].get('evidence'),
                        'suggestion_for_improvements': data[0].get('suggestion_for_improvements')
                    }
                    st.download_button(
                        label='🚨 Download Bias Detection Results',
                        data=json.dumps(bias_data, indent=4),
                        file_name='bias_detection.json',
                        mime='application/json'
                    )
            
            # Executive Summary
            # st.subheader("📊 Executive Summary")
            
#             # Create a comprehensive summary
#             data_summary = data[0]
#             summary = {
#                 'analysis_timestamp': pd.Timestamp.now().isoformat(),
#                 'analysis_complete': True,
#                 'total_reviews_analyzed': len(data),
#                 'sentiment_summary': format_list_data(data_summary, 'sentiment', 'Not analyzed'),
#                 'tone_summary': format_list_data(data_summary, 'tone', 'Not analyzed'),
#                 'consistency_summary': format_list_data(data_summary, 'consistency', 'Not analyzed'),
#                 'bias_detected_summary': format_list_data(data_summary, 'bias_detected', 'Not analyzed'),
#                 'bias_type_summary': format_bias_type_display(data_summary.get('bias_type', 'None')),
#             }

#             # Add average confidence if available
#             confidence_data = data_summary.get('confidence_score', [])
#             if isinstance(confidence_data, list) and confidence_data and all(isinstance(c, (int, float)) for c in confidence_data):
#                 summary['avg_confidence_score'] = sum(confidence_data) / len(confidence_data)
#             else:
#                 summary['confidence_score'] = confidence_data
            
#             st.json(summary)
            
#         else:
#             st.info("🔄 Complete the analysis to download results")

# else:
#     st.info("📁 Please upload a CSV or JSON file to begin analysis")
#     st.markdown("""
#     ### How to use this app:
#     1. **Upload File**: Use the sidebar to upload your CSV or JSON file containing review data
#     2. **Start Analysis**: Click the "Start Complete Analysis" button to run all analysis modules
#     3. **View Results**: Navigate through the tabs to see different analysis results
#     4. **Download**: Use the Download tab to save your results in various formats
    
#     ### Expected File Format:
#     Your JSON file should contain review data with a 'review_contents' field containing the reviews to analyze.
    
#     ### Data Structure:
#     This app handles both single values and lists of values for all analysis results, displaying distributions and summaries as appropriate.
#     """)
                # Executive Summary
            st.subheader("📊 Executive Summary")

            # Create a comprehensive summary
            data_summary = data[0]

            # Calculate summary statistics
            bias_detected = data_summary.get('bias_detected', [])
            confidence_scores = data_summary.get('confidence_score', [])

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>📋 Analysis Overview</h4>
                </div>
                """, unsafe_allow_html=True)
                
                total_reviews = len(data) if isinstance(data, list) else 1
                st.metric("Total Reviews Analyzed", total_reviews)
                st.metric("Analysis Status", "✅ Complete")

            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>😊 Sentiment & Tone</h4>
                </div>
                """, unsafe_allow_html=True)
                
                sentiment_summary = format_list_data(data_summary, 'sentiment', 'Not analyzed')
                tone_summary = format_list_data(data_summary, 'tone', 'Not analyzed')
                st.write(f"**Sentiment:** {sentiment_summary}")
                st.write(f"**Tone:** {tone_summary}")

            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h4>🚨 Bias Detection</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # bias_summary = format_list_data(data_summary, 'bias_detected', 'Not analyzed')
                # if isinstance(bias_detected, list) and any(bias_detected):
                #     st.error(f"⚠️ Bias Detected")
                # else:
                #     st.success("✅ No Bias Detected")

                # Check for actual bias (excluding "None")
                bias_type_data = data_summary.get('bias_type', [])
                has_actual_bias = False

                if isinstance(bias_type_data, list):
                    for bias_types in bias_type_data:
                        if isinstance(bias_types, list):
                            if any(bt and str(bt).lower() != 'none' for bt in bias_types):
                                has_actual_bias = True
                                break
                        elif bias_types and str(bias_types).lower() != 'none':
                            has_actual_bias = True
                            break
                elif bias_type_data and str(bias_type_data).lower() != 'none':
                    has_actual_bias = True

                if has_actual_bias:
                    st.error("⚠️ Bias Detected")
                else:
                    st.success("✅ No Bias Detected")

            # Detailed Summary Cards
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                # Consistency Summary
                st.markdown("""
                <div class="metric-card">
                    <h4>🔄 Consistency Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                
                consistency_summary = format_list_data(data_summary, 'consistency', 'Not analyzed')
                st.write(f"**Internal Consistency:** {consistency_summary}")
                
                alignment_summary = format_list_data(data_summary, 'is_consistent_with_others', 'Not analyzed')
                st.write(f"**Inter-Review Alignment:** {alignment_summary}")

            with col2:
                # Confidence & Quality Metrics
                st.markdown("""
                <div class="metric-card">
                    <h4>📊 Quality Metrics</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Average confidence calculation
                if isinstance(confidence_scores, list) and confidence_scores and all(isinstance(c, (int, float)) for c in confidence_scores):
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                    st.metric("Average Confidence Score", f"{avg_confidence:.2f}")
                elif isinstance(confidence_scores, (int, float)):
                    st.metric("Confidence Score", f"{confidence_scores:.2f}")
                
                # Bias type summary
                bias_type_summary = format_bias_type_display(data_summary.get('bias_type', 'None'))
                if isinstance(bias_type_summary, list):
                    bias_types_str = ", ".join([bt for bt in bias_type_summary if bt != "None"])
                    st.write(f"**Bias Types:** {bias_types_str if bias_types_str else 'None detected'}")
                else:
                    st.write(f"**Bias Types:** {bias_type_summary}")

            # Key Findings Summary
            st.markdown("---")
            st.subheader("🔍 Key Findings")

            findings = []

            # Add findings based on analysis results
            if 'contradictory_points' in data_summary and data_summary['contradictory_points']:
                contradictions = data_summary['contradictory_points']
                contradiction_count = len(contradictions) if isinstance(contradictions, list) else 1
                findings.append(f"⚠️ Found {contradiction_count} contradictory point(s) between reviews")

            # if isinstance(bias_detected, list) and any(bias_detected):
            #     biased_count = sum(1 for b in bias_detected if b)
            #     findings.append(f"🚨 Bias detected in {biased_count} out of {len(bias_detected)} reviews")

            if isinstance(bias_detected, list):
                # Count biased reviews excluding "None" bias types
                bias_type_data = data_summary.get('bias_type', [])
                biased_count = 0
                
                if isinstance(bias_type_data, list):
                    for bias_types in bias_type_data:
                        if isinstance(bias_types, list):
                            # Check if any bias type in the list is not "None"
                            if any(bt and str(bt).lower() != 'none' for bt in bias_types):
                                biased_count += 1
                        elif bias_types and str(bias_types).lower() != 'none':
                            biased_count += 1
                
                if biased_count > 0:
                    findings.append(f"🚨 Bias detected in {biased_count} out of {len(bias_detected)} reviews")

            if 'possible_bias_flags' in data_summary and data_summary['possible_bias_flags']:
                flags = data_summary['possible_bias_flags']
                flag_count = len(flags) if isinstance(flags, list) else 1
                findings.append(f"🚩 {flag_count} possible bias flag(s) identified")

            if not findings:
                findings.append("✅ No significant issues detected in the review analysis")

            for finding in findings:
                st.write(finding)

            # Analysis timestamp
            st.markdown("---")
            st.caption(f"Analysis completed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")