"""Report generation in multiple formats"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports in JSON, HTML, and PDF formats"""
    
    @staticmethod
    def to_json(report: Dict[str, Any], output_path: str = None) -> str:
        """Export report as JSON"""
        if output_path is None:
            output_path = f"report_{report.get('analysis_id', 'unknown')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_path
    
    @staticmethod
    def to_html(report: Dict[str, Any]) -> str:
        """Generate HTML report"""
        detection = report.get("detection_result", {})
        overall = report.get("overall_risk", {})
        
        risk_color = {
            "CRITICAL": "#dc3545",
            "HIGH": "#fd7e14",
            "MEDIUM": "#ffc107",
            "LOW": "#20c997",
            "MINIMAL": "#28a745",
            "UNKNOWN": "#6c757d"
        }.get(overall.get("level", "UNKNOWN"), "#6c757d")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DeepGuard AI Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ border-bottom: 3px solid #1a73e8; padding-bottom: 20px; margin-bottom: 30px; }}
                .risk-badge {{ display: inline-block; padding: 10px 25px; border-radius: 5px; color: white; font-weight: bold; font-size: 1.2em; background: {risk_color}; }}
                .section {{ margin: 25px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .section h2 {{ margin-top: 0; color: #1a73e8; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #1a73e8; color: white; }}
                .finding {{ padding: 10px; margin: 5px 0; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; }}
                .recommendation {{ padding: 10px; margin: 5px 0; background: #d4edda; border-left: 4px solid #28a745; border-radius: 4px; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛡️ DeepGuard AI Analysis Report</h1>
                    <p><strong>Analysis ID:</strong> {report.get('analysis_id', 'N/A')}</p>
                    <p><strong>Timestamp:</strong> {report.get('timestamp', 'N/A')}</p>
                </div>
                
                <div class="section">
                    <h2>Overall Risk Assessment</h2>
                    <p><span class="risk-badge">{overall.get('level', 'UNKNOWN')} RISK</span></p>
                    <p><strong>Risk Score:</strong> {overall.get('score', 0):.2%}</p>
                    <p><strong>Threat Detected:</strong> {'⚠️ YES' if overall.get('is_threat') else '✅ NO'}</p>
                    <p><strong>Confidence:</strong> {overall.get('confidence', 0):.2%}</p>
                </div>
                
                <div class="section">
                    <h2>File Information</h2>
                    <table>
                        <tr><th>Property</th><th>Value</th></tr>
                        <tr><td>File Name</td><td>{report.get('file_info', {}).get('name', 'N/A')}</td></tr>
                        <tr><td>File Type</td><td>{report.get('file_info', {}).get('type', 'N/A')}</td></tr>
                        <tr><td>File Size</td><td>{report.get('file_info', {}).get('size_mb', 0)} MB</td></tr>
                        <tr><td>SHA-256 Hash</td><td style="font-family: monospace; font-size: 0.8em;">{report.get('file_info', {}).get('sha256', 'N/A')}</td></tr>
                        <tr><td>Processing Time</td><td>{report.get('performance', {}).get('processing_time_seconds', 0)} seconds</td></tr>
                        <tr><td>Integrity Preserved</td><td>{'✅ Yes' if report.get('integrity', {}).get('integrity_preserved') else '⚠️ No'}</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Detection Findings</h2>
                    <h3>Indicators Found:</h3>
                    {''.join(f'<div class="finding">⚠️ {f}</div>' for f in detection.get('findings', ['None detected']))}
                    
                    <h3>Recommendations:</h3>
                    {''.join(f'<div class="recommendation">✅ {r}</div>' for r in detection.get('recommendations', ['No action needed']))}
                </div>
                
                <div class="footer">
                    <p>Generated by DeepGuard AI v1.0.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>⚠️ This report is an automated analysis and should be reviewed by a security professional before taking action.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    @staticmethod
    def to_pdf(report: Dict[str, Any], output_path: str = None) -> str:
        """Generate PDF report using reportlab"""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        
        if output_path is None:
            output_path = f"report_{report.get('analysis_id', 'unknown')}.pdf"
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'],
            fontSize=24, textColor=HexColor('#1a73e8'), spaceAfter=30
        )
        story.append(Paragraph("DeepGuard AI Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Overview
        overall = report.get("overall_risk", {})
        story.append(Paragraph(f"Analysis ID: {report.get('analysis_id', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"Timestamp: {report.get('timestamp', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"Risk Level: {overall.get('level', 'UNKNOWN')}", styles['Normal']))
        story.append(Paragraph(f"Risk Score: {overall.get('score', 0):.2%}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # File info table
        file_info = report.get("file_info", {})
        data = [
            ["Property", "Value"],
            ["File Name", file_info.get("name", "N/A")],
            ["File Type", file_info.get("type", "N/A")],
            ["File Size", f"{file_info.get('size_mb', 0)} MB"],
            ["SHA-256", file_info.get("sha256", "N/A")[:32] + "..."],
            ["Processing Time", f"{report.get('performance', {}).get('processing_time_seconds', 0)}s"],
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a73e8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#ddd')),
        ]))
        story.append(table)
        
        doc.build(story)
        return output_path