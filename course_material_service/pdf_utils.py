import io
from markdown import markdown
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
import re

def markdown_to_pdf(markdown_text: str, title: str = "Course Material") -> bytes:
    """
    Converts Markdown text to a PDF file in memory.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title=title
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=TA_JUSTIFY, leading=14, fontSize=11, fontName='Times-Roman'))
    styles.add(ParagraphStyle(name='Quote', parent=styles['Normal'], leftIndent=20, rightIndent=20, textColor=colors.gray, fontName='Times-Italic'))
    
    # Custom Heading Styles
    styles['Heading1'].fontName = 'Times-Bold'
    styles['Heading1'].fontSize = 24
    styles['Heading1'].leading = 28
    styles['Heading1'].spaceAfter = 12
    
    styles['Heading2'].fontName = 'Times-Bold'
    styles['Heading2'].fontSize = 18
    styles['Heading2'].leading = 22
    styles['Heading2'].spaceBefore = 12
    styles['Heading2'].spaceAfter = 10

    styles['Heading3'].fontName = 'Times-Bold'
    styles['Heading3'].fontSize = 14
    styles['Heading3'].leading = 18
    styles['Heading3'].spaceBefore = 10
    styles['Heading3'].spaceAfter = 6

    # Convert Markdown to a simpler format we can parse line-by-line or block-by-block
    # Note: A full HTML-to-ReportLab parser is complex. 
    # We will do a heuristic parsing of the Markdown directly for common elements.
    
    story = []
    
    # Title
    story.append(Paragraph(title, styles['Heading1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Alpha Learning Systems", styles['Normal']))
    story.append(Spacer(1, 24))

    # Split by double newlines to get paragraphs/blocks
    blocks = markdown_text.split('\n\n')
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # Headers
        if block.startswith('# '):
            story.append(Paragraph(block[2:], styles['Heading1']))
        elif block.startswith('## '):
            story.append(Paragraph(block[3:], styles['Heading2']))
        elif block.startswith('### '):
            story.append(Paragraph(block[4:], styles['Heading3']))
            
        # Blockquotes
        elif block.startswith('> '):
            clean_text = block.replace('> ', '').replace('\n', ' ')
            story.append(Paragraph(clean_text, styles['Quote']))
            story.append(Spacer(1, 8))
            
        # Lists (simple handling)
        elif block.startswith('* ') or block.startswith('- '):
            items = []
            for line in block.split('\n'):
                if line.startswith('* ') or line.startswith('- '):
                    clean = line[2:].strip()
                    # Handle bolding **text** -> <b>text</b> for ReportLab
                    clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean)
                    items.append(ListItem(Paragraph(clean, styles['Normal'])))
            story.append(ListFlowable(items, bulletType='bullet', start='circle'))
            story.append(Spacer(1, 8))
            
        # Normal Paragraphs
        else:
            # Handle bolding **text** -> <b>text</b>
            clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', block)
            # Handle italics *text* -> <i>text</i>
            clean = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean)
            # Replace newlines with spaces for flow
            clean = clean.replace('\n', ' ')
            story.append(Paragraph(clean, styles['Justify']))
            story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
