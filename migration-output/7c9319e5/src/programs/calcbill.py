"""CALCBILL -- Bill calculation subprogram.

Migrated from: lema01/CALCBILL (PROGRAM-ID. CALCBILL)

Called as a subprogram via COBOL CALL.  LINKAGE SECTION parameters
become function arguments.

CRITICAL: Stock check uses >= (greater-or-equal), NOT >.
          ORDCOMP uses >.  These are intentionally different.

BILL STATUS values (from source):
    "INVALID PRODUCT" -- product not found (AT END, WS-FOUND-FLAG=0)
    "NOT AVAILABLE"   -- product found but QOH is zero
    "NOT SUFFICIENT"  -- product found, QOH > 0 but QOH < quantity_required

After shipping, CALCBILL does REWRITE MASTERFILE-REC to update QOH in-place.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

from models.product import CalcbillMasterRecord
from utils.file_io import read_records, SequentialFileWriter

# Module-level file handles -- equivalent to WS-OPEN-FLAG lazy open.
# CALCBILL opens shipping/pending files on first call and keeps them
# open across multiple calls until close_flag=1.
_pend_writer: SequentialFileWriter | None = None
_ship_writer: SequentialFileWriter | None = None
_pend_fh: TextIO | None = None
_ship_fh: TextIO | None = None


def calcbill(
    transaction_id: str,
    product_id: str,
    quantity_required: int,
    close_flag: int,
) -> None:
    """Entry point -- equivalent to 0000-MAIN-PARA.

    LINKAGE SECTION parameters:
        WS-TRANSACTION-ID      -> transaction_id
        WS-PRODUCT-ID          -> product_id
        WS-QUANTITYREQUIRED    -> quantity_required
        WS-CLOSE-FLAG          -> close_flag (1 = close files, 0 = process)

    File paths from env vars (JCL DD assignments):
        INFILE1   -- master product file
        OUTFILE1  -- pending orders output
        OUTFILE2  -- shipping orders output
    """
    global _pend_fh, _ship_fh

    if close_flag == 1:
        # Close shipping and pending files
        _close_output_files()
        return

    master_path = os.environ.get("INFILE1", "data/masterfile.dat")
    pend_path = os.environ.get("OUTFILE1", "data/pendingfile.dat")
    ship_path = os.environ.get("OUTFILE2", "data/shippingfile.dat")

    # Lazy open -- WS-OPEN-FLAG pattern
    _ensure_output_files_open(pend_path, ship_path)

    # 1000-OPEN-READ-PARA + 2000-MASTER-READ-PARA
    # Read all master records for potential REWRITE
    master_lines = _read_file_lines(master_path)
    found = False

    for i, line in enumerate(master_lines):
        master = CalcbillMasterRecord.from_line(line)

        if master.product_id == product_id:
            found = True

            if master.quantity_on_hand >= quantity_required:
                # COBOL checks >= first (ship branch)
                bill = master.price * quantity_required

                ship_line = _format_shipping_line(
                    transaction_id, product_id, quantity_required, bill
                )
                _ship_fh.write(ship_line + "\n")

                # REWRITE MASTERFILE-REC: reduce QOH and update master in-place
                master.quantity_on_hand -= quantity_required
                master_lines[i] = master.to_line()
                _rewrite_file(master_path, master_lines)
            elif master.quantity_on_hand == 0:
                # COBOL checks IS ZERO inside ELSE -- NOT AVAILABLE
                pend_line = _format_pending_line(
                    transaction_id, product_id, quantity_required, "NOT AVAILABLE"
                )
                _pend_fh.write(pend_line + "\n")
            else:
                # QOH > 0 but < quantity_required -- NOT SUFFICIENT
                pend_line = _format_pending_line(
                    transaction_id, product_id, quantity_required, "NOT SUFFICIENT"
                )
                _pend_fh.write(pend_line + "\n")
            break

    if not found:
        # Product not found in master -- INVALID PRODUCT
        pend_line = _format_pending_line(
            transaction_id, product_id, quantity_required, "INVALID PRODUCT"
        )
        _pend_fh.write(pend_line + "\n")


def _ensure_output_files_open(pend_path: str, ship_path: str) -> None:
    """Lazy-open output files on first call (WS-OPEN-FLAG pattern)."""
    global _pend_fh, _ship_fh

    if _pend_fh is None:
        os.makedirs(os.path.dirname(pend_path) or ".", exist_ok=True)
        _pend_fh = open(pend_path, "w", encoding="utf-8")

    if _ship_fh is None:
        os.makedirs(os.path.dirname(ship_path) or ".", exist_ok=True)
        _ship_fh = open(ship_path, "w", encoding="utf-8")


def _close_output_files() -> None:
    """Close output files -- called when close_flag=1."""
    global _pend_fh, _ship_fh

    if _ship_fh is not None:
        _ship_fh.close()
        _ship_fh = None

    if _pend_fh is not None:
        _pend_fh.close()
        _pend_fh = None


def _read_file_lines(path: str) -> list[str]:
    """Read all lines from a fixed-width file, preserving line content."""
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n").rstrip("\r") for line in fh if line.strip()]


def _rewrite_file(path: str, lines: list[str]) -> None:
    """REWRITE pattern: rewrite entire file after updating a record.

    Equivalent to COBOL REWRITE MASTERFILE-REC.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


def _format_pending_line(
    tran_id: str, product_id: str, qty: int, status: str
) -> str:
    """Format a CALCBILL pending record (PENDINGREC) as 80-char line.

    CALCBILL PENDINGREC layout (all DISPLAY, no COMP-3):
        TRANSACTIONID  PIC X(02)
        PRODUCTID      PIC X(04)
        QUANTITYREQUIRED PIC 9(03)
        BILLSTATUS     PIC X(15)
        FILLER         PIC X(56)
    """
    parts = [
        tran_id.ljust(2)[:2],
        product_id.ljust(4)[:4],
        str(qty).zfill(3)[:3],
        status.ljust(15)[:15],
    ]
    return "".join(parts).ljust(80)[:80]


def _format_shipping_line(
    tran_id: str, product_id: str, qty: int, bill: int
) -> str:
    """Format a CALCBILL shipping record (SHIPPINGREC) as 80-char line.

    CALCBILL SHIPPINGREC layout (all DISPLAY, no COMP-3):
        TRANSACTIONID    PIC X(02)
        PRODUCTID        PIC X(04)
        QUANTITYREQUIRED PIC 9(03)
        TOTALBILL        PIC 9(05)
        FILLER           PIC X(66)
    """
    parts = [
        tran_id.ljust(2)[:2],
        product_id.ljust(4)[:4],
        str(qty).zfill(3)[:3],
        str(bill).zfill(5)[:5],
    ]
    return "".join(parts).ljust(80)[:80]
