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

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import markdown
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import base64
from reportlab.lib.units import inch
from reportlab.platypus import Image as ReportLabImage
import io
from collections import Counter

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'manual_reviews' not in st.session_state:
    st.session_state.manual_reviews = [""]  # Start with one empty review

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

def process_csv_data(df):
    """Convert CSV DataFrame to JSON format expected by the analysis"""
    # Define expected column mappings
    review_columns = ['review', 'review_content', 'review_text', 'content', 'text']
    title_columns = ['paper_title', 'title', 'paper_name', 'name']
    abstract_columns = ['paper_abstract', 'abstract', 'summary', 'description']
    
    # Find the review content column
    review_column = None
    for col in df.columns:
        if col.lower() in review_columns:
            review_column = col
            break
    
    if review_column is None:
        # If no standard column found, use the first text column
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            review_column = text_columns[0]
        else:
            raise ValueError("No suitable text column found for reviews")
    
    # Find paper title column (optional)
    title_column = None
    for col in df.columns:
        if col.lower() in title_columns:
            title_column = col
            break
    
    # Find paper abstract column (optional)
    abstract_column = None
    for col in df.columns:
        if col.lower() in abstract_columns:
            abstract_column = col
            break
    
    # Get the first row's title and abstract (assuming all rows have same paper)
    paper_title = df[title_column].iloc[0] if title_column and not df[title_column].isna().iloc[0] else None
    paper_abstract = df[abstract_column].iloc[0] if abstract_column and not df[abstract_column].isna().iloc[0] else None
    
    # Convert to the expected JSON format
    review_contents = df[review_column].dropna().tolist()
    
    json_data = [{
        'review_contents': review_contents,
        'source': 'csv_upload',
        'total_reviews': len(review_contents)
    }]
    
    # Add paper info if available
    if paper_title:
        json_data[0]['paper_title'] = paper_title
    if paper_abstract:
        json_data[0]['paper_abstract'] = paper_abstract
    
    return json_data

def create_manual_input_data(paper_title, paper_abstract, reviews):
    """Convert manual input to JSON format expected by the analysis"""
    # Filter out empty reviews
    filtered_reviews = [review.strip() for review in reviews if review.strip()]
    
    if not filtered_reviews:
        raise ValueError("Please enter at least one review")
    
    json_data = [{
        'paper_title': paper_title,
        'paper_abstract': paper_abstract,
        'review_contents': filtered_reviews,
        'source': 'manual_input',
        'total_reviews': len(filtered_reviews)
    }]
    
    return json_data

# def generate_pdf_report(data):
#     """Generate a comprehensive analysis report in PDF format"""
#     # Generate markdown content first
#     markdown_content = generate_comprehensive_report(data)
    
#     # Create a BytesIO buffer
#     buffer = BytesIO()
    
#     # Create the PDF document
#     doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
#                           topMargin=72, bottomMargin=18)
    
#     # Get styles
#     styles = getSampleStyleSheet()
    
#     # Custom styles
#     title_style = ParagraphStyle(
#         'CustomTitle',
#         parent=styles['Heading1'],
#         fontSize=18,
#         spaceAfter=30,
#         alignment=TA_CENTER,
#         textColor=colors.darkblue
#     )
    
#     heading_style = ParagraphStyle(
#         'CustomHeading',
#         parent=styles['Heading2'],
#         fontSize=14,
#         spaceAfter=12,
#         spaceBefore=20,
#         textColor=colors.darkred
#     )
    
#     subheading_style = ParagraphStyle(
#         'CustomSubHeading',
#         parent=styles['Heading3'],
#         fontSize=12,
#         spaceAfter=8,
#         spaceBefore=12,
#         textColor=colors.black
#     )
    
#     normal_style = styles['Normal']
    
#     # Build story
#     story = []
    
#     # Process markdown content line by line
#     lines = markdown_content.split('\n')
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             story.append(Spacer(1, 6))
#             continue
            
#         # Title
#         if line.startswith('# '):
#             title_text = line[2:].replace('📊', '').strip()
#             story.append(Paragraph(title_text, title_style))
#             story.append(Spacer(1, 12))
            
#         # Main headings
#         elif line.startswith('## '):
#             heading_text = line[3:].replace('📄', '').replace('📋', '').replace('😊', '').replace('🔄', '').replace('⚖️', '').replace('🚨', '').replace('📊', '').strip()
#             story.append(Paragraph(heading_text, heading_style))
#             story.append(Spacer(1, 8))
            
#         # Sub headings
#         elif line.startswith('### '):
#             subheading_text = line[4:].replace('⚠️', '').replace('🚩', '').strip()
#             story.append(Paragraph(subheading_text, subheading_style))
#             story.append(Spacer(1, 6))
            
#         # Bold text (key-value pairs)
#         elif line.startswith('**') and line.endswith('**'):
#             clean_text = line[2:-2]
#             story.append(Paragraph(f"<b>{clean_text}</b>", normal_style))
            
#         # List items
#         elif line.startswith('- **'):
#             # Extract key-value format
#             if ':**' in line:
#                 parts = line.split(':**', 1)
#                 key = parts[0][3:].strip()  # Remove '- **'
#                 value = parts[1].strip() if len(parts) > 1 else ''
#                 story.append(Paragraph(f"<b>{key}:</b> {value}", normal_style))
#             else:
#                 clean_text = line[2:].replace('**', '').replace('📊', '').replace('✅', '').replace('⚠️', '').replace('🚨', '').strip()
#                 story.append(Paragraph(f"• {clean_text}", normal_style))
                
#         # Regular list items
#         elif line.startswith('- '):
#             clean_text = line[2:].replace('**', '').replace('📊', '').replace('✅', '').replace('⚠️', '').replace('🚨', '').strip()
#             story.append(Paragraph(f"• {clean_text}", normal_style))
            
#         # Numbered lists
#         elif line and line[0].isdigit() and line[1:3] == '. ':
#             clean_text = line.replace('**', '').replace('📊', '').replace('✅', '').replace('⚠️', '').replace('🚨', '').strip()
#             story.append(Paragraph(clean_text, normal_style))
            
#         # Horizontal rules
#         elif line.startswith('---'):
#             story.append(Spacer(1, 12))
            
#         # Regular paragraphs
#         elif line and not line.startswith('*'):
#             clean_text = line.replace('**', '').replace('📊', '').replace('✅', '').replace('⚠️', '').replace('🚨', '').replace('📄', '').replace('📋', '').replace('😊', '').replace('🔄', '').replace('⚖️', '').strip()
#             if clean_text:
#                 story.append(Paragraph(clean_text, normal_style))
#                 story.append(Spacer(1, 6))
    
#     # Build PDF
#     doc.build(story)
    
#     # Get the value of the BytesIO buffer
#     pdf_data = buffer.getvalue()
#     buffer.close()
    
#     return pdf_data

# Add these imports at the top of your file (add to existing imports)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import base64
from reportlab.lib.units import inch
from reportlab.platypus import Image as ReportLabImage
import io
from collections import Counter

# Add this function to generate charts for PDF (add this new function)
def generate_charts_for_pdf(data):
    """Generate charts and return them as base64 encoded images for PDF inclusion"""
    charts = {}
    
    # Set style for better looking charts
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Sentiment Distribution Chart
    sentiment_data = data.get('sentiment', [])
    if sentiment_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if isinstance(sentiment_data, list):
            sentiment_counts = Counter(sentiment_data)
        else:
            sentiment_counts = Counter([sentiment_data])
        
        colors = ['#2E8B57', '#FFD700', '#FF6347', '#4682B4', '#DDA0DD']
        wedges, texts, autotexts = ax.pie(sentiment_counts.values(), 
                                         labels=sentiment_counts.keys(), 
                                         autopct='%1.1f%%',
                                         colors=colors[:len(sentiment_counts)],
                                         startangle=90)
        
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        charts['sentiment_chart'] = buffer
        plt.close()
    
    # 2. Tone Distribution Chart
    tone_data = data.get('tone', [])
    if tone_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if isinstance(tone_data, list):
            tone_counts = Counter(tone_data)
        else:
            tone_counts = Counter([tone_data])
        
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        wedges, texts, autotexts = ax.pie(tone_counts.values(), 
                                         labels=tone_counts.keys(), 
                                         autopct='%1.1f%%',
                                         colors=colors[:len(tone_counts)],
                                         startangle=90)
        
        ax.set_title('Tone Distribution', fontsize=14, fontweight='bold')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        charts['tone_chart'] = buffer
        plt.close()
    
    # 3. Consistency Distribution Chart
    consistency_data = data.get('consistency', [])
    if consistency_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if isinstance(consistency_data, list):
            consistency_counts = Counter(consistency_data)
        else:
            consistency_counts = Counter([consistency_data])
        
        colors = ['#90EE90', '#FFB6C1', '#87CEEB']
        wedges, texts, autotexts = ax.pie(consistency_counts.values(), 
                                         labels=consistency_counts.keys(), 
                                         autopct='%1.1f%%',
                                         colors=colors[:len(consistency_counts)],
                                         startangle=90)
        
        ax.set_title('Consistency Distribution', fontsize=14, fontweight='bold')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        charts['consistency_chart'] = buffer
        plt.close()
    
    # 4. Confidence Scores Chart
    confidence_scores = data.get('confidence_score', [])
    if confidence_scores and isinstance(confidence_scores, list) and len(confidence_scores) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = range(len(confidence_scores))
        bars = ax.bar(x_pos, confidence_scores, color='skyblue', alpha=0.7, edgecolor='navy')
        
        ax.set_xlabel('Review Number', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Confidence Scores Across Reviews', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Review {i+1}' for i in x_pos])
        
        # Add value labels on bars
        for bar, score in zip(bars, confidence_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(confidence_scores) * 1.1)
        ax.grid(True, alpha=0.3)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        charts['confidence_chart'] = buffer
        plt.close()
    
    # 5. Alignment Scores Chart
    alignment_scores = data.get('alignment_score', [])
    if alignment_scores and isinstance(alignment_scores, list) and len(alignment_scores) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = range(len(alignment_scores))
        bars = ax.bar(x_pos, alignment_scores, color='lightcoral', alpha=0.7, edgecolor='darkred')
        
        ax.set_xlabel('Review Pair', fontsize=12)
        ax.set_ylabel('Alignment Score', fontsize=12)
        ax.set_title('Alignment Scores Across Review Pairs', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Pair {i+1}' for i in x_pos])
        
        # Add value labels on bars
        for bar, score in zip(bars, alignment_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(alignment_scores) * 1.1)
        ax.grid(True, alpha=0.3)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        charts['alignment_chart'] = buffer
        plt.close()
    
    # 6. Bias Type Distribution Chart
    bias_type_data = data.get('bias_type', [])
    if bias_type_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Flatten bias types if nested
        all_bias_types = []
        if isinstance(bias_type_data, list):
            for bias_types in bias_type_data:
                if isinstance(bias_types, list):
                    all_bias_types.extend([bt for bt in bias_types if bt and str(bt).lower() != 'none'])
                elif bias_types and str(bias_types).lower() != 'none':
                    all_bias_types.append(bias_types)
        elif bias_type_data and str(bias_type_data).lower() != 'none':
            all_bias_types.append(bias_type_data)
        
        if all_bias_types:
            bias_counts = Counter(all_bias_types)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            wedges, texts, autotexts = ax.pie(bias_counts.values(), 
                                             labels=bias_counts.keys(), 
                                             autopct='%1.1f%%',
                                             colors=colors[:len(bias_counts)],
                                             startangle=90)
            
            ax.set_title('Bias Types Distribution', fontsize=14, fontweight='bold')
        else:
            # No bias detected - show a simple message
            ax.text(0.5, 0.5, 'No Bias Detected', ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='green')
            ax.set_title('Bias Types Distribution', fontsize=14, fontweight='bold')
        
        ax.axis('equal')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        charts['bias_chart'] = buffer
        plt.close()
    
    return charts

# Replace the existing generate_pdf_report function with this enhanced version
def generate_pdf_report(data):
    """Generate a comprehensive analysis report in PDF format with charts"""
    # Generate charts first
    charts = generate_charts_for_pdf(data[0])
    
    # Generate markdown content
    markdown_content = generate_comprehensive_report(data)
    
    # Create a BytesIO buffer
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkred
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.black
    )
    
    normal_style = styles['Normal']
    
    # Build story
    story = []
    
    # Process markdown content line by line
    lines = markdown_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
            
        # Title
        if line.startswith('# '):
            title_text = line[2:].replace('📊', '').strip()
            story.append(Paragraph(title_text, title_style))
            story.append(Spacer(1, 12))
            
        # Main headings - Add charts after specific sections
        elif line.startswith('## '):
            heading_text = line[3:].replace('📄', '').replace('📋', '').replace('😊', '').replace('🔄', '').replace('⚖️', '').replace('🚨', '').replace('📊', '').strip()
            story.append(Paragraph(heading_text, heading_style))
            story.append(Spacer(1, 8))
            
            # Add charts after specific headings
            if 'Sentiment' in heading_text and 'Tone' in heading_text:
                # Add sentiment and tone charts
                if 'sentiment_chart' in charts:
                    story.append(Spacer(1, 12))
                    story.append(ReportLabImage(charts['sentiment_chart'], width=4*inch, height=3*inch))
                    story.append(Spacer(1, 6))
                
                if 'tone_chart' in charts:
                    story.append(Spacer(1, 6))
                    story.append(ReportLabImage(charts['tone_chart'], width=4*inch, height=3*inch))
                    story.append(Spacer(1, 12))
                    
            elif 'Consistency' in heading_text:
                # Add consistency chart
                if 'consistency_chart' in charts:
                    story.append(Spacer(1, 12))
                    story.append(ReportLabImage(charts['consistency_chart'], width=4*inch, height=3*inch))
                    story.append(Spacer(1, 12))
                    
            elif 'Inter-Review Comparison' in heading_text:
                # Add alignment chart
                if 'alignment_chart' in charts:
                    story.append(Spacer(1, 12))
                    story.append(ReportLabImage(charts['alignment_chart'], width=5*inch, height=3*inch))
                    story.append(Spacer(1, 12))
                    
            elif 'Bias Detection' in heading_text:
                # Add confidence and bias charts
                if 'confidence_chart' in charts:
                    story.append(Spacer(1, 12))
                    story.append(ReportLabImage(charts['confidence_chart'], width=5*inch, height=3*inch))
                    story.append(Spacer(1, 6))
                
                if 'bias_chart' in charts:
                    story.append(Spacer(1, 6))
                    story.append(ReportLabImage(charts['bias_chart'], width=4*inch, height=3*inch))
                    story.append(Spacer(1, 12))
            
        # Sub headings
        elif line.startswith('### '):
            subheading_text = line[4:].replace('⚠️', '').replace('🚩', '').strip()
            story.append(Paragraph(subheading_text, subheading_style))
            story.append(Spacer(1, 6))
            
        # Bold text (key-value pairs)
        elif line.startswith('**') and line.endswith('**'):
            clean_text = line[2:-2]
            story.append(Paragraph(f"<b>{clean_text}</b>", normal_style))
            
        # List items
        elif line.startswith('- **'):
            # Extract key-value format
            if ':**' in line:
                parts = line.split(':**', 1)
                key = parts[0][3:].strip()  # Remove '- **'
                value = parts[1].strip() if len(parts) > 1 else ''
                story.append(Paragraph(f"<b>{key}:</b> {value}", normal_style))
            else:
                clean_text = line[2:].replace('**', '').replace('📊', '').replace('✅', '').replace('⚠️', '').replace('🚨', '').strip()
                story.append(Paragraph(f"• {clean_text}", normal_style))
                
        # Regular list items
        elif line.startswith('- '):
            clean_text = line[2:].replace('**', '').replace('📊', '').replace('✅', '').replace('⚠️', '').replace('🚨', '').strip()
            story.append(Paragraph(f"• {clean_text}", normal_style))
            
        # Numbered lists
        elif line and line[0].isdigit() and line[1:3] == '. ':
            clean_text = line.replace('**', '').replace('📊', '').replace('✅', '').replace('⚠️', '').replace('🚨', '').strip()
            story.append(Paragraph(clean_text, normal_style))
            
        # Horizontal rules
        elif line.startswith('---'):
            story.append(Spacer(1, 12))
            
        # Regular paragraphs
        elif line and not line.startswith('*'):
            clean_text = line.replace('**', '').replace('📊', '').replace('✅', '').replace('⚠️', '').replace('🚨', '').replace('📄', '').replace('📋', '').replace('😊', '').replace('🔄', '').replace('⚖️', '').strip()
            if clean_text:
                story.append(Paragraph(clean_text, normal_style))
                story.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(story)
    
    # Close chart buffers
    for chart_buffer in charts.values():
        chart_buffer.close()
    
    # Get the value of the BytesIO buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def generate_comprehensive_report(data):
    """Generate a comprehensive analysis report in markdown format"""
    print(data)
    report_data = data[0]
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Start building the report
    report = f"""# 📊 Comprehensive Review Analysis Report

Generated on: {timestamp}  
Analysis Status: Complete
Total Reviews Analyzed: {len(report_data.get('review_contents', []))}

---

"""
    
    report += "---\n\n"
    
    # Executive Summary
    report += "## 📋 Executive Summary\n\n"
    
    # Calculate summary statistics
    bias_detected = report_data.get('bias_detected', [])
    bias_type_data = report_data.get('bias_type', [])
    confidence_scores = report_data.get('confidence_score', [])
    
    # Check for actual bias
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
    
    # Add summary metrics
    report += f"- **Overall Sentiment:** {format_list_data(report_data, 'sentiment', 'Not analyzed')}\n"
    report += f"- **Overall Tone:** {format_list_data(report_data, 'tone', 'Not analyzed')}\n"
    report += f"- **Internal Consistency:** {format_list_data(report_data, 'consistency', 'Not analyzed')}\n"
    report += f"- **Inter-Review Alignment:** {format_list_data(report_data, 'is_consistent_with_others', 'Not analyzed')}\n"
    report += f"- **Bias Detection Status:** {'⚠️ Bias Detected' if has_actual_bias else '✅ No Bias Detected'}\n"
    
    # Average confidence
    if isinstance(confidence_scores, list) and confidence_scores and all(isinstance(c, (int, float)) for c in confidence_scores):
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        report += f"- **Average Confidence Score:** {avg_confidence:.2f}\n"
    
    report += "\n---\n\n"
    
    # Detailed Analysis Sections
    
    # 1. Sentiment and Tone Analysis
    report += "## 😊 Sentiment & Tone Analysis\n\n"
    
    sentiment_data = report_data.get('sentiment', [])
    tone_data = report_data.get('tone', [])
    sentiment_reason_data = report_data.get('sentiment_reason', [])
    tone_reason_data = report_data.get('tone_reason', [])
    
    if isinstance(sentiment_data, list) and len(sentiment_data) > 1:
        for i in range(len(sentiment_data)):
            report += f"### Review {i+1}\n"
            report += f"- **Sentiment:** {sentiment_data[i] if i < len(sentiment_data) else 'N/A'}\n"
            report += f"- **Tone:** {tone_data[i] if i < len(tone_data) else 'N/A'}\n"
            
            if isinstance(sentiment_reason_data, list) and i < len(sentiment_reason_data):
                report += f"- **Sentiment Analysis:** {sentiment_reason_data[i]}\n"
            if isinstance(tone_reason_data, list) and i < len(tone_reason_data):
                report += f"- **Tone Analysis:** {tone_reason_data[i]}\n"
            report += "\n"
    else:
        report += f"- **Overall Sentiment:** {sentiment_data}\n"
        report += f"- **Overall Tone:** {tone_data}\n"
        if sentiment_reason_data:
            report += f"- **Sentiment Analysis:** {sentiment_reason_data}\n"
        if tone_reason_data:
            report += f"- **Tone Analysis:** {tone_reason_data}\n"
    
    report += "\n---\n\n"
    
    # 2. Consistency Analysis
    report += "## 🔄 Consistency Analysis\n\n"
    
    consistency_data = report_data.get('consistency', [])
    consistency_reason_data = report_data.get('consistency_reason', [])
    
    if isinstance(consistency_data, list) and len(consistency_data) > 1:
        for i in range(len(consistency_data)):
            report += f"### Review {i+1}\n"
            report += f"- **Consistency Status:** {consistency_data[i] if i < len(consistency_data) else 'N/A'}\n"
            if isinstance(consistency_reason_data, list) and i < len(consistency_reason_data):
                report += f"- **Analysis:** {consistency_reason_data[i]}\n"
            report += "\n"
    else:
        report += f"- **Overall Consistency:** {consistency_data}\n"
        if consistency_reason_data:
            report += f"- **Analysis:** {consistency_reason_data}\n"
    
    report += "\n---\n\n"
    
    # 3. Inter-Review Comparison
    report += "## ⚖️ Inter-Review Comparison Analysis\n\n"
    
    report += f"- **Overall Alignment:** {format_list_data(report_data, 'is_consistent_with_others', 'Not analyzed')}\n"
    
    alignment_scores = report_data.get('alignment_score', [])
    if isinstance(alignment_scores, list) and alignment_scores:
        avg_alignment = sum(alignment_scores) / len(alignment_scores)
        report += f"- **Average Alignment Score:** {avg_alignment:.2f}\n"
    else:
        report += f"- **Alignment Score:** {alignment_scores}\n"
    
    # Contradictory points
    contradictions = report_data.get('contradictory_points', [])
    if contradictions:
        report += "\n### ⚠️ Contradictory Points:\n"
        if isinstance(contradictions, list):
            for i, point in enumerate(contradictions):
                report += f"{i+1}. {point}\n"
        else:
            report += f"- {contradictions}\n"
    
    # Bias flags
    bias_flags = report_data.get('possible_bias_flags', [])
    if bias_flags:
        report += "\n### 🚩 Possible Bias Flags:\n"
        if isinstance(bias_flags, list):
            for i, flag in enumerate(bias_flags):
                report += f"{i+1}. {flag}\n"
        else:
            report += f"- {bias_flags}\n"
    
    report += "\n---\n\n"
    
    # 4. Bias Detection Results
    report += "## 🚨 Bias Detection Results\n\n"
    
    evidence_data = report_data.get('evidence', [])
    suggestions_data = report_data.get('suggestion_for_improvements', [])
    
    if isinstance(bias_detected, list) and len(bias_detected) > 1:
        for i in range(len(bias_detected)):
            report += f"### Review {i+1}\n"
            
            # Bias status
            review_bias = bias_detected[i] if i < len(bias_detected) else False
            report += f"- **Bias Detected:** {'Yes' if review_bias else 'No'}\n"
            
            # Bias type
            review_bias_type = bias_type_data[i] if isinstance(bias_type_data, list) and i < len(bias_type_data) else bias_type_data
            formatted_bias_type = format_bias_type_display(review_bias_type)
            if isinstance(formatted_bias_type, list):
                bias_types_str = ", ".join([bt for bt in formatted_bias_type if bt != "None"])
                report += f"- **Bias Type(s):** {bias_types_str if bias_types_str else 'None'}\n"
            else:
                report += f"- **Bias Type:** {formatted_bias_type}\n"
            
            # Confidence
            review_confidence = confidence_scores[i] if isinstance(confidence_scores, list) and i < len(confidence_scores) else confidence_scores
            report += f"- **Confidence Score:** {review_confidence}\n"
            
            # Evidence
            if isinstance(evidence_data, list) and i < len(evidence_data) and evidence_data[i]:
                report += f"- **Evidence:** {evidence_data[i]}\n"
            
            # Suggestions
            if isinstance(suggestions_data, list) and i < len(suggestions_data) and suggestions_data[i]:
                report += f"- **Improvement Suggestions:** {suggestions_data[i]}\n"
            
            report += "\n"
    else:
        report += f"- **Bias Detected:** {'Yes' if bias_detected else 'No'}\n"
        
        formatted_bias_type = format_bias_type_display(bias_type_data)
        if isinstance(formatted_bias_type, list):
            bias_types_str = ", ".join([bt for bt in formatted_bias_type if bt != "None"])
            report += f"- **Bias Type(s):** {bias_types_str if bias_types_str else 'None'}\n"
        else:
            report += f"- **Bias Type:** {formatted_bias_type}\n"
        
        report += f"- **Confidence Score:** {confidence_scores}\n"
        
        if evidence_data:
            report += f"- **Evidence:** {evidence_data}\n"
        if suggestions_data:
            report += f"- **Improvement Suggestions:** {suggestions_data}\n"
    
    report += "\n---\n\n"
    
    # Raw Data Summary
    report += "## 📊 Raw Data Summary\n\n"
    report += f"- **Data Source:** {report_data.get('source', 'JSON').replace('_', ' ').title()}\n"
    report += f"- **Total Reviews:** {len(report_data.get('review_contents', []))}\n"
    report += f"- **Analysis Modules:** Sentiment Analysis, Consistency Check, Inter-Review Comparison, Bias Detection\n"
    
    report += "\n---\n\n"
    report += "*Report generated by Review Bias Detection & Analysis System*"
    
    return report

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
    
    .manual-input-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
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

# Sidebar for file upload and manual input
with st.sidebar:
    st.header("📥 Data Input Options")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["File Upload", "Manual Input"],
        index=0
    )

    if input_method == "File Upload":
        st.subheader("📁 File Upload")
        file = st.file_uploader(
            label="Upload files (CSV, JSON):", 
            accept_multiple_files=False, 
            type=['csv', 'json']
        )
        
        # ADD THIS FORMAT INFORMATION BLOCK HERE
        with st.expander("📋 Expected File Formats", expanded=False):
            st.markdown("""
            **CSV Format Requirements:**
            - Must contain a column with review text (any of these names):
            - `review`, `review_content`, `review_text`, `content`, `text`
            - Optional columns:
            - `paper_title`, `title`, `paper_name`, `name` (for paper title)
            - `paper_abstract`, `abstract`, `summary`, `description` (for paper abstract)
            
            **Example CSV:**
            ```
            review,paper_title,paper_abstract
            "This paper presents excellent methodology...",Paper Title,Abstract text here
            "The analysis is thorough but lacks...",Paper Title,Abstract text here
            ```
            
            **JSON Format Requirements:**
            ```json
            [{
                "review_contents": [
                    "Review 1 text here...",
                    "Review 2 text here...",
                    "Review 3 text here..."
                ],
                "paper_title": "Optional paper title",
                "paper_abstract": "Optional paper abstract"
            }]
            ```
            """)
        
        # Process uploaded file
        if file:
            try:
                if file.name.split('.')[-1] == 'csv':
                    file_type = 'csv'
                    uploaded_df = pd.read_csv(file)
                    st.session_state.data = process_csv_data(uploaded_df)
                    
                    st.success(f"✅ {file.name} uploaded successfully!")
                    with st.expander("📊 CSV Preview", expanded=False):
                        st.dataframe(uploaded_df.head())
                        st.write(f"**Rows:** {len(uploaded_df)}")
                        st.write(f"**Columns:** {', '.join(uploaded_df.columns)}")
                        
                        # Show detected paper info
                        if 'paper_title' in st.session_state.data[0]:
                            st.write(f"**Paper Title:** {st.session_state.data[0]['paper_title']}")
                        if 'paper_abstract' in st.session_state.data[0]:
                            st.write(f"**Paper Abstract:** {st.session_state.data[0]['paper_abstract'][:100]}...")
                        
                elif file.name.split('.')[-1] == 'json':
                    file_type = 'json'
                    st.session_state.data = json.load(file)
                    st.success(f"✅ {file.name} uploaded successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error uploading file: {str(e)}")
                st.info("**CSV files should have:**\n- Review text column: 'review', 'content', 'text'\n- Optional: 'paper_title', 'paper_abstract'")
    
    else:  # Manual Input
        st.subheader("✏️ Manual Input")
        
        # Paper details
        paper_title = st.text_input(
            "📄 Paper Title:",
            placeholder="Enter the title of the paper being reviewed"
        )
        
        paper_abstract = st.text_area(
            "📝 Paper Abstract:",
            placeholder="Enter the abstract of the paper (optional)",
            height=100
        )
        
        st.markdown("---")
        st.write("**📋 Review Contents:**")
        
        # Dynamic review input fields
        for i in range(len(st.session_state.manual_reviews)):
            review_text = st.text_area(
                f"Review {i+1}:",
                value=st.session_state.manual_reviews[i],
                key=f"review_{i}",
                placeholder=f"Enter review {i+1} content here...",
                height=100
            )
            st.session_state.manual_reviews[i] = review_text
        
        # Buttons for adding/removing review fields
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Review", key="add_review"):
                st.session_state.manual_reviews.append("")
                st.rerun()
        
        with col2:
            if len(st.session_state.manual_reviews) > 1:
                if st.button("➖ Remove Last", key="remove_review"):
                    st.session_state.manual_reviews.pop()
                    st.rerun()
        
        # Process manual input
        if st.button("💾 Use Manual Input", type="primary"):
            try:
                st.session_state.data = create_manual_input_data(
                    paper_title, 
                    paper_abstract, 
                    st.session_state.manual_reviews
                )
                st.success("✅ Manual input processed successfully!")
                st.info(f"📊 Processed {len([r for r in st.session_state.manual_reviews if r.strip()])} reviews")
            except ValueError as e:
                st.error(f"❌ {str(e)}")

    if st.session_state.data is not None:
        st.markdown("---")
        st.header("🔍 Analysis Controls")
        
        # Add warning message here
        st.warning("⏳ **Analysis Time Notice:** Complete analysis in the last stage may take 2-3 minutes per review. So, the total time depends on the number of reviews and complexity. Please be patient while all modules are processed.")
        
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
            st.metric("Total Reviews", len(data[0]['review_contents']))
        with col2:
            source = data[0].get('source', 'JSON')
            st.metric("Data Source", 'JSON' if source=='JSON' else source.replace('_', ' ').title())
        with col3:
            st.metric("Analysis Status", "✅ Complete" if st.session_state.analysis_complete else "⏳ Pending")
        
        # Show paper details if available (from manual input)
        if 'paper_title' in data[0] and data[0]['paper_title']:
            st.subheader("📄 Paper Information")
            st.write(f"**Title:** {data[0]['paper_title']}")
            if data[0].get('paper_abstract'):
                with st.expander("📝 Abstract", expanded=False):
                    st.write(data[0]['paper_abstract'])
        
        # Display raw data
        with st.expander("🔍 View Raw Data", expanded=False):
            st.json(data[0])
        
        # Review contents preview
        st.subheader("📝 Review Contents Preview")
        for i, item in enumerate(data[0]['review_contents']):
            with st.expander(f"Review {i+1}", expanded=False):
                st.write(item)
    
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


        # with tab6:
        #     st.header("💾 Export & Download Results")
            
        #     if st.session_state.analysis_complete:
        #         data = st.session_state.data
                
        #         st.subheader("📄 Comprehensive Analysis Report")

        #         # Generate reports
        #         comprehensive_report_md = generate_comprehensive_report(data)
        #         comprehensive_report_pdf = generate_pdf_report(data)

        #         col1, col2 = st.columns(2)
        #         with col1:
        #             st.download_button(
        #                 label='📄 Download Report (Markdown)',
        #                 data=comprehensive_report_md,
        #                 file_name=f'analysis_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.md',
        #                 mime='text/markdown',
        #                 help="Download analysis report in Markdown format"
        #             )

        #         with col2:
        #             st.download_button(
        #                 label='📋 Download Report (PDF)',
        #                 data=comprehensive_report_pdf,
        #                 file_name=f'analysis_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pdf',
        #                 mime='application/pdf',
        #                 help="Download analysis report in PDF format"
        #             )
                
        #         st.markdown("---")
                
        #         # Full analysis download (existing code continues...)
        #         download_data = json.dumps(data, indent=4)
        #         st.download_button(
        #             label='📥 Download Complete Analysis (JSON)',
        #             data=download_data,
        #             file_name='complete_reviews_analysis.json',
        #             mime='application/json'
        #         )            
        #     # Individual analysis downloads with improved layout
        #     st.subheader("📋 Individual Component Downloads")
            
        #     col1, col2 = st.columns(2)
            
        #     with col1:
        #         # Sentiment analysis only
        #         if 'sentiment' in data[0]:
        #             sentiment_data = {
        #                 'sentiment': data[0].get('sentiment'),
        #                 'sentiment_reason': data[0].get('sentiment_reason'),
        #                 'tone': data[0].get('tone'),
        #                 'tone_reason': data[0].get('tone_reason')
        #             }
        #             st.download_button(
        #                 label='😊 Download Sentiment Analysis',
        #                 data=json.dumps(sentiment_data, indent=4),
        #                 file_name='sentiment_analysis.json',
        #                 mime='application/json'
        #             )
                
        #         # Consistency analysis only
        #         if 'consistency' in data[0]:
        #             consistency_data = {
        #                 'consistency': data[0].get('consistency'),
        #                 'consistency_reason': data[0].get('consistency_reason')
        #             }
        #             st.download_button(
        #                 label='🔄 Download Consistency Analysis',
        #                 data=json.dumps(consistency_data, indent=4),
        #                 file_name='consistency_analysis.json',
        #                 mime='application/json'
        #             )
            
        #     with col2:
        #         # Inter-review comparison only
        #         if 'alignment_score' in data[0]:
        #             comparison_data = {
        #                 'is_consistent_with_others': data[0].get('is_consistent_with_others'),
        #                 'alignment_score': data[0].get('alignment_score'),
        #                 'contradictory_points': data[0].get('contradictory_points'),
        #                 'possible_bias_flags': data[0].get('possible_bias_flags'),
        #                 'summary_of_differences': data[0].get('summary_of_differences')
        #             }
        #             st.download_button(
        #                 label='⚖️ Download Inter-Review Comparison',
        #                 data=json.dumps(comparison_data, indent=4),
        #                 file_name='inter_review_comparison.json',
        #                 mime='application/json'
        #             )
                
        #         # Bias detection only
        #         if 'bias_detected' in data[0]:
        #             bias_data = {
        #                 'bias_detected': data[0].get('bias_detected'),
        #                 'bias_type': data[0].get('bias_type'),
        #                 'confidence_score': data[0].get('confidence_score'),
        #                 'evidence': data[0].get('evidence'),
        #                 'suggestion_for_improvements': data[0].get('suggestion_for_improvements')
        #             }
        #             st.download_button(
        #                 label='🚨 Download Bias Detection Results',
        #                 data=json.dumps(bias_data, indent=4),
        #                 file_name='bias_detection.json',
        #                 mime='application/json'
        #             )

        with tab6:
            st.header("💾 Export & Download Results")
            
            if st.session_state.analysis_complete:
                data = st.session_state.data
                
                st.subheader("📄 All-in-One Download Package")
                
                # Generate all reports
                comprehensive_report_md = generate_comprehensive_report(data)
                comprehensive_report_pdf = generate_pdf_report(data)
                complete_analysis_json = json.dumps(data, indent=4)
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                
                # Create ZIP file in memory
                import zipfile
                from io import BytesIO
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add Markdown report
                    zip_file.writestr(f'analysis_report_{timestamp}.md', comprehensive_report_md)
                    
                    # Add PDF report
                    zip_file.writestr(f'analysis_report_{timestamp}.pdf', comprehensive_report_pdf)
                    
                    # Add complete JSON analysis
                    zip_file.writestr(f'complete_analysis_{timestamp}.json', complete_analysis_json)
                    
                    # Add individual analysis components
                    sentiment_data = {
                        'sentiment': data[0].get('sentiment', 'Not analyzed'),
                        'sentiment_reason': data[0].get('sentiment_reason', 'Not analyzed'),
                        'tone': data[0].get('tone', 'Not analyzed'),
                        'tone_reason': data[0].get('tone_reason', 'Not analyzed')
                    }
                    zip_file.writestr(f'sentiment_analysis_{timestamp}.json', json.dumps(sentiment_data, indent=4))
                    
                    consistency_data = {
                        'consistency': data[0].get('consistency', 'Not analyzed'),
                        'consistency_reason': data[0].get('consistency_reason', 'Not analyzed')
                    }
                    zip_file.writestr(f'consistency_analysis_{timestamp}.json', json.dumps(consistency_data, indent=4))
                    
                    comparison_data = {
                        'is_consistent_with_others': data[0].get('is_consistent_with_others', 'Not analyzed'),
                        'alignment_score': data[0].get('alignment_score', 'Not analyzed'),
                        'contradictory_points': data[0].get('contradictory_points', 'Not analyzed'),
                        'possible_bias_flags': data[0].get('possible_bias_flags', 'Not analyzed'),
                        'summary_of_differences': data[0].get('summary_of_differences', 'Not analyzed')
                    }
                    zip_file.writestr(f'comparison_analysis_{timestamp}.json', json.dumps(comparison_data, indent=4))
                    
                    bias_data = {
                        'bias_detected': data[0].get('bias_detected', 'Not analyzed'),
                        'bias_type': data[0].get('bias_type', 'Not analyzed'),
                        'confidence_score': data[0].get('confidence_score', 'Not analyzed'),
                        'evidence': data[0].get('evidence', 'Not analyzed'),
                        'suggestion_for_improvements': data[0].get('suggestion_for_improvements', 'Not analyzed')
                    }
                    zip_file.writestr(f'bias_analysis_{timestamp}.json', json.dumps(bias_data, indent=4))
                
                zip_buffer.seek(0)
                
                # Single download button for everything
                st.download_button(
                    label='📦 Download Complete Analysis Package (ZIP)',
                    data=zip_buffer.getvalue(),
                    file_name=f'review_analysis_complete_{timestamp}.zip',
                    mime='application/zip',
                    help="Downloads ZIP file containing: Markdown report, PDF report, complete JSON analysis, and individual component analyses",
                    type="primary"
                )
                
                # Show what's included
                with st.expander("📋 Package Contents", expanded=False):
                    st.markdown("""
                    **This ZIP package includes:**
                    - 📄 Analysis Report (Markdown format)
                    - 📋 Analysis Report (PDF format) 
                    - 📥 Complete Analysis Data (JSON)
                    - 😊 Sentiment Analysis (JSON)
                    - 🔄 Consistency Analysis (JSON)
                    - ⚖️ Inter-Review Comparison (JSON)
                    - 🚨 Bias Detection Results (JSON)
                    """)
                
                st.markdown("---")
                
                # # Optional: Keep individual downloads as backup
                # st.subheader("📋 Individual Downloads (Optional)")
                
                # col1, col2, col3 = st.columns(3)
                # with col1:
                #     st.download_button(
                #         label='📄 Markdown Report',
                #         data=comprehensive_report_md,
                #         file_name=f'report_{timestamp}.md',
                #         mime='text/markdown'
                #     )
                # with col2:
                #     st.download_button(
                #         label='📋 PDF Report',
                #         data=comprehensive_report_pdf,
                #         file_name=f'report_{timestamp}.pdf',
                #         mime='application/pdf'
                #     )
                # with col3:
                #     st.download_button(
                #         label='📥 JSON Data',
                #         data=complete_analysis_json,
                #         file_name=f'analysis_{timestamp}.json',
                #         mime='application/json'
                #     )
        
        # Rest of your existing executive summary code...
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