"""Analytics services module for Excel dashboard generation.

This package provides services for analyzing uploaded data files and generating
analytical insights. It supports Excel and CSV file parsing, data profiling
using YData Profiler, KPI analysis, and dashboard generation.

Available Services:
- ExcelParserService: Parse and normalize Excel/CSV files
- YDataProfilerService: Generate comprehensive data profiles
- KPIAnalyzerService: Extract and analyze key performance indicators
- DashboardGeneratorService: Generate interactive dashboards
"""
from .excel_parser import ExcelParserService
from .ydata_profiler import YDataProfilerService
from .kpi_analyzer import KPIAnalyzerService
from .dashboard_generator import DashboardGeneratorService

__all__ = [
    "ExcelParserService",
    "YDataProfilerService",
    "KPIAnalyzerService",
    "DashboardGeneratorService",
]
