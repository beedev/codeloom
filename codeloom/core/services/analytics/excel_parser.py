"""Excel parser service for analytics module.

This module provides functionality to parse Excel and CSV files, normalize
data types, and extract metadata for downstream analytics processing.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


class ExcelParserService:
    """Service for parsing and normalizing Excel/CSV files.

    Handles file parsing, data type inference, date parsing, missing value
    handling, and metadata extraction. Supports both Excel (.xlsx, .xls)
    and CSV file formats.

    Attributes:
        logger: Logger instance for operation tracking.
    """

    # Supported file extensions
    SUPPORTED_EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}
    SUPPORTED_CSV_EXTENSIONS = {".csv", ".tsv"}
    SUPPORTED_EXTENSIONS = SUPPORTED_EXCEL_EXTENSIONS | SUPPORTED_CSV_EXTENSIONS

    # Common date column name patterns for auto-detection
    DATE_COLUMN_PATTERNS = [
        "date", "time", "datetime", "timestamp", "created", "updated",
        "modified", "period", "month", "year", "day", "week", "quarter"
    ]

    def __init__(self) -> None:
        """Initialize the Excel parser service."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO)

    @property
    def logger(self) -> logging.Logger:
        """Access to the service logger."""
        return self._logger

    def parse(
        self,
        file: Union[bytes, str, Path],
        sheet_name: Optional[Union[str, int]] = None,
        infer_types: bool = True,
        parse_dates: bool = True,
        handle_missing: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse an Excel or CSV file and return normalized DataFrame with metadata.

        Args:
            file: File content as bytes, file path as string, or Path object.
            sheet_name: Optional sheet name or index for Excel files.
                       Uses first sheet if None. Ignored for CSV files.
            infer_types: Whether to infer and convert data types. Defaults to True.
            parse_dates: Whether to parse date columns. Defaults to True.
            handle_missing: Whether to handle missing values. Defaults to True.

        Returns:
            Tuple containing:
            - pd.DataFrame: Parsed and normalized DataFrame
            - Dict[str, Any]: Metadata dictionary containing:
                - row_count: Number of rows in the DataFrame
                - column_count: Number of columns
                - column_info: List of column metadata dictionaries
                - file_type: Detected file type (excel/csv)
                - sheet_name: Name of parsed sheet (Excel only)
                - available_sheets: List of available sheets (Excel only)
                - missing_value_summary: Summary of missing values
                - parsing_warnings: List of any warnings during parsing

        Raises:
            ValueError: If file format is unsupported or file is empty.
            FileNotFoundError: If file path does not exist.
            pd.errors.EmptyDataError: If file contains no data.
            Exception: For other parsing errors with descriptive message.
        """
        self._log_operation("parse", file_type=type(file).__name__)

        try:
            # Determine file type and read DataFrame
            df, file_metadata = self._read_file(file, sheet_name)

            # Validate DataFrame is not empty
            if df.empty:
                raise ValueError("File contains no data or all columns are empty")

            # Track parsing warnings
            warnings: List[str] = []

            # Normalize column names
            df, name_warnings = self._normalize_column_names(df)
            warnings.extend(name_warnings)

            # Infer and convert data types
            if infer_types:
                df, type_warnings = self._infer_data_types(df, parse_dates)
                warnings.extend(type_warnings)

            # Handle missing values
            missing_summary: Dict[str, Any] = {}
            if handle_missing:
                df, missing_summary, missing_warnings = self._handle_missing_values(df)
                warnings.extend(missing_warnings)

            # Build column info metadata
            column_info = self._build_column_info(df)

            # Build complete metadata
            metadata: Dict[str, Any] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "column_info": column_info,
                "file_type": file_metadata.get("file_type", "unknown"),
                "sheet_name": file_metadata.get("sheet_name"),
                "available_sheets": file_metadata.get("available_sheets"),
                "missing_value_summary": missing_summary,
                "parsing_warnings": warnings,
            }

            self.logger.info(
                f"Successfully parsed file: {metadata['row_count']} rows, "
                f"{metadata['column_count']} columns"
            )

            return df, metadata

        except FileNotFoundError:
            self._log_error("parse", "File not found")
            raise
        except pd.errors.EmptyDataError:
            self._log_error("parse", "File contains no data")
            raise ValueError("File contains no data")
        except Exception as e:
            self._log_error("parse", str(e))
            raise

    def get_sheet_names(self, file: Union[bytes, str, Path]) -> List[str]:
        """Get list of sheet names from an Excel file.

        Args:
            file: File content as bytes, file path as string, or Path object.

        Returns:
            List of sheet names. Returns empty list for CSV files.

        Raises:
            ValueError: If file format is unsupported.
            FileNotFoundError: If file path does not exist.
        """
        file_type, file_obj = self._prepare_file_object(file)

        if file_type == "csv":
            return []

        try:
            excel_file = pd.ExcelFile(file_obj)
            return excel_file.sheet_names
        except Exception as e:
            self._log_error("get_sheet_names", str(e))
            raise ValueError(f"Failed to read sheet names: {e}")

    def validate_file(self, file: Union[bytes, str, Path]) -> Dict[str, Any]:
        """Validate a file before parsing.

        Performs lightweight validation without fully parsing the file.

        Args:
            file: File content as bytes, file path as string, or Path object.

        Returns:
            Dictionary containing:
            - valid: Boolean indicating if file is valid
            - file_type: Detected file type
            - error: Error message if invalid
            - sheet_count: Number of sheets (Excel only)
        """
        result: Dict[str, Any] = {
            "valid": False,
            "file_type": None,
            "error": None,
            "sheet_count": None,
        }

        try:
            file_type, file_obj = self._prepare_file_object(file)
            result["file_type"] = file_type

            if file_type == "excel":
                excel_file = pd.ExcelFile(file_obj)
                result["sheet_count"] = len(excel_file.sheet_names)
            else:
                # For CSV, just check if we can read first few rows
                pd.read_csv(file_obj, nrows=5)

            result["valid"] = True

        except Exception as e:
            result["error"] = str(e)

        return result

    def _read_file(
        self,
        file: Union[bytes, str, Path],
        sheet_name: Optional[Union[str, int]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read file and return DataFrame with file metadata.

        Args:
            file: File content or path.
            sheet_name: Sheet name or index for Excel files.

        Returns:
            Tuple of (DataFrame, file metadata dict).

        Raises:
            ValueError: If file format is unsupported.
            FileNotFoundError: If file path does not exist.
        """
        file_type, file_obj = self._prepare_file_object(file)
        metadata: Dict[str, Any] = {"file_type": file_type}

        if file_type == "excel":
            df, excel_meta = self._read_excel(file_obj, sheet_name)
            metadata.update(excel_meta)
        else:
            df = self._read_csv(file_obj)
            metadata["sheet_name"] = None
            metadata["available_sheets"] = None

        return df, metadata

    def _prepare_file_object(
        self,
        file: Union[bytes, str, Path]
    ) -> Tuple[str, Union[io.BytesIO, str, Path]]:
        """Prepare file object for reading and detect file type.

        Args:
            file: File content as bytes, file path, or Path object.

        Returns:
            Tuple of (file_type, file_object) where file_type is 'excel' or 'csv'.

        Raises:
            ValueError: If file format cannot be determined or is unsupported.
            FileNotFoundError: If file path does not exist.
        """
        if isinstance(file, bytes):
            # Try to detect file type from magic bytes
            file_type = self._detect_file_type_from_bytes(file)
            return file_type, io.BytesIO(file)

        # Handle string or Path
        file_path = Path(file) if isinstance(file, str) else file

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()

        if extension in self.SUPPORTED_EXCEL_EXTENSIONS:
            return "excel", file_path
        elif extension in self.SUPPORTED_CSV_EXTENSIONS:
            return "csv", file_path
        else:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

    def _detect_file_type_from_bytes(self, file_bytes: bytes) -> str:
        """Detect file type from file content bytes.

        Args:
            file_bytes: File content as bytes.

        Returns:
            File type string: 'excel' or 'csv'.

        Raises:
            ValueError: If file type cannot be determined.
        """
        # Check for Excel magic bytes (ZIP format for xlsx, OLE for xls)
        # XLSX/XLSM/XLSB files start with PK (ZIP format)
        if file_bytes[:2] == b'PK':
            return "excel"
        # XLS files start with OLE compound document signature
        if file_bytes[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
            return "excel"

        # Assume CSV for text-based content
        # Try to decode as text to verify it's readable
        try:
            # Check first 1000 bytes for text content
            sample = file_bytes[:1000].decode('utf-8', errors='strict')
            # If it contains common CSV delimiters, assume CSV
            if ',' in sample or '\t' in sample or ';' in sample:
                return "csv"
        except UnicodeDecodeError:
            pass

        raise ValueError(
            "Could not determine file type from content. "
            "Please provide a file path with extension or valid Excel/CSV bytes."
        )

    def _read_excel(
        self,
        file_obj: Union[io.BytesIO, str, Path],
        sheet_name: Optional[Union[str, int]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read Excel file and return DataFrame with metadata.

        Args:
            file_obj: File object or path.
            sheet_name: Sheet name or index. Uses first sheet if None.

        Returns:
            Tuple of (DataFrame, metadata dict).
        """
        excel_file = pd.ExcelFile(file_obj)
        available_sheets = excel_file.sheet_names

        # Determine which sheet to read
        if sheet_name is None:
            target_sheet = available_sheets[0] if available_sheets else 0
        else:
            target_sheet = sheet_name

        # Validate sheet exists
        if isinstance(target_sheet, str) and target_sheet not in available_sheets:
            raise ValueError(
                f"Sheet '{target_sheet}' not found. "
                f"Available sheets: {', '.join(available_sheets)}"
            )
        if isinstance(target_sheet, int) and target_sheet >= len(available_sheets):
            raise ValueError(
                f"Sheet index {target_sheet} out of range. "
                f"File has {len(available_sheets)} sheets."
            )

        df = pd.read_excel(excel_file, sheet_name=target_sheet)

        # Get actual sheet name if index was provided
        actual_sheet_name = (
            available_sheets[target_sheet]
            if isinstance(target_sheet, int)
            else target_sheet
        )

        metadata = {
            "sheet_name": actual_sheet_name,
            "available_sheets": available_sheets,
        }

        return df, metadata

    def _read_csv(self, file_obj: Union[io.BytesIO, str, Path]) -> pd.DataFrame:
        """Read CSV file with encoding detection.

        Args:
            file_obj: File object or path.

        Returns:
            Parsed DataFrame.
        """
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                # Reset file position if BytesIO
                if isinstance(file_obj, io.BytesIO):
                    file_obj.seek(0)

                df = pd.read_csv(file_obj, encoding=encoding)
                return df

            except UnicodeDecodeError:
                continue
            except Exception as e:
                # For non-encoding errors, raise immediately
                if "codec" not in str(e).lower():
                    raise

        raise ValueError(
            f"Could not decode CSV file with any supported encoding: {encodings}"
        )

    def _normalize_column_names(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Normalize column names for consistency.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (DataFrame with normalized columns, list of warnings).
        """
        warnings: List[str] = []
        new_columns: List[str] = []
        seen_names: Dict[str, int] = {}

        for col in df.columns:
            # Convert to string and strip whitespace
            name = str(col).strip()

            # Replace problematic characters
            name = name.replace('\n', ' ').replace('\r', ' ')

            # Handle empty column names
            if not name or name.lower() == 'unnamed':
                name = f"column_{len(new_columns)}"
                warnings.append(f"Renamed empty/unnamed column to '{name}'")

            # Handle duplicate column names
            if name in seen_names:
                seen_names[name] += 1
                new_name = f"{name}_{seen_names[name]}"
                warnings.append(f"Renamed duplicate column '{name}' to '{new_name}'")
                name = new_name
            else:
                seen_names[name] = 0

            new_columns.append(name)

        df.columns = new_columns
        return df, warnings

    def _infer_data_types(
        self,
        df: pd.DataFrame,
        parse_dates: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Infer and convert data types for each column.

        Args:
            df: Input DataFrame.
            parse_dates: Whether to parse date columns.

        Returns:
            Tuple of (DataFrame with inferred types, list of warnings).
        """
        warnings: List[str] = []
        df = df.copy()

        for col in df.columns:
            original_dtype = df[col].dtype

            # Skip if already a non-object type
            if original_dtype != object:
                continue

            # Try numeric conversion first
            numeric_result = pd.to_numeric(df[col], errors='coerce')
            non_null_original = df[col].notna().sum()
            non_null_numeric = numeric_result.notna().sum()

            # If most values convert successfully, use numeric
            if non_null_numeric >= non_null_original * 0.9:
                df[col] = numeric_result
                continue

            # Try datetime conversion if enabled
            if parse_dates and self._is_likely_date_column(col, df[col]):
                try:
                    date_result = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    non_null_date = date_result.notna().sum()

                    if non_null_date >= non_null_original * 0.8:
                        df[col] = date_result
                        continue
                except Exception:
                    pass

            # Try boolean conversion
            bool_result = self._try_boolean_conversion(df[col])
            if bool_result is not None:
                df[col] = bool_result
                continue

            # Keep as string/object if no conversion applies
            # But ensure consistent string type
            df[col] = df[col].astype(str).replace('nan', pd.NA)

        return df, warnings

    def _is_likely_date_column(self, column_name: str, series: pd.Series) -> bool:
        """Check if column is likely to contain dates.

        Args:
            column_name: Name of the column.
            series: Column data.

        Returns:
            True if column likely contains dates.
        """
        # Check column name
        col_lower = column_name.lower()
        for pattern in self.DATE_COLUMN_PATTERNS:
            if pattern in col_lower:
                return True

        # Check sample values for date patterns
        sample = series.dropna().head(10).astype(str)
        date_pattern_count = 0

        for val in sample:
            # Common date patterns: contains / or - with numbers
            if any(sep in val for sep in ['/', '-']) and any(c.isdigit() for c in val):
                date_pattern_count += 1

        return date_pattern_count >= len(sample) * 0.5

    def _try_boolean_conversion(self, series: pd.Series) -> Optional[pd.Series]:
        """Try to convert series to boolean type.

        Args:
            series: Input series.

        Returns:
            Boolean series if conversion successful, None otherwise.
        """
        # Common boolean representations
        true_values = {'true', 'yes', '1', 'y', 't', 'on'}
        false_values = {'false', 'no', '0', 'n', 'f', 'off'}

        # Get unique non-null values
        unique_vals = set(series.dropna().astype(str).str.lower().unique())

        # Check if all values are boolean-like
        if unique_vals <= (true_values | false_values):
            result = series.astype(str).str.lower()
            result = result.isin(true_values)
            return result.where(series.notna(), pd.NA)

        return None

    def _handle_missing_values(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        """Handle and summarize missing values.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (DataFrame, missing value summary, warnings).
        """
        warnings: List[str] = []
        df = df.copy()

        # Calculate missing value statistics
        missing_counts = df.isna().sum()
        total_rows = len(df)

        summary: Dict[str, Any] = {
            "total_missing_cells": int(df.isna().sum().sum()),
            "total_cells": int(df.size),
            "missing_percentage": round(
                df.isna().sum().sum() / df.size * 100, 2
            ) if df.size > 0 else 0,
            "columns_with_missing": {},
        }

        for col in df.columns:
            missing_count = int(missing_counts[col])
            if missing_count > 0:
                missing_pct = round(missing_count / total_rows * 100, 2)
                summary["columns_with_missing"][col] = {
                    "count": missing_count,
                    "percentage": missing_pct,
                }

                # Warn about high missing percentages
                if missing_pct > 50:
                    warnings.append(
                        f"Column '{col}' has {missing_pct}% missing values"
                    )

        # Standardize missing value representation
        # Convert various null representations to pandas NA
        null_representations = ['', 'null', 'NULL', 'None', 'none', 'N/A', 'n/a', 'NA', 'na', '#N/A', '-']

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].replace(null_representations, pd.NA)

        return df, summary, warnings

    def _build_column_info(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build detailed column information metadata.

        Args:
            df: Input DataFrame.

        Returns:
            List of column metadata dictionaries.
        """
        column_info: List[Dict[str, Any]] = []

        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)

            info: Dict[str, Any] = {
                "name": col,
                "dtype": dtype,
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "unique_count": int(series.nunique()),
            }

            # Add type-specific statistics
            if pd.api.types.is_numeric_dtype(series):
                info.update({
                    "min": float(series.min()) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                    "mean": float(series.mean()) if series.notna().any() else None,
                    "std": float(series.std()) if series.notna().any() else None,
                })
            elif pd.api.types.is_datetime64_any_dtype(series):
                info.update({
                    "min": str(series.min()) if series.notna().any() else None,
                    "max": str(series.max()) if series.notna().any() else None,
                })
            else:
                # String/categorical column
                if series.notna().any():
                    info["sample_values"] = series.dropna().head(5).tolist()

            column_info.append(info)

        return column_info

    def _log_operation(self, operation: str, **kwargs: Any) -> None:
        """Log a service operation with context.

        Args:
            operation: Operation name.
            **kwargs: Additional context to log.
        """
        context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self._logger.info(f"{operation}: {context}")

    def _log_error(self, operation: str, error: str) -> None:
        """Log an error with context.

        Args:
            operation: Operation that failed.
            error: Error message.
        """
        self._logger.error(f"{operation} failed: {error}")
