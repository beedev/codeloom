"""VSAMORD -- VSAM-based order processing program.

Migrated from: VSAMORD (PROGRAM-ID. VSAMORD)

Key differences from sequential programs:
- Master file is INDEXED (VSAM KSDS) with RANDOM access
- Uses READ MASTERIN KEY IS PRODUCT-ID (random key lookup, not sequential scan)
- Master PRODUCT-ID is PIC X(05) (5 chars, not 4)
- Transaction PRODID is PIC X(05) (5 chars)
- Stock check: QOH >= QTY-REQ (greater-or-equal)
- Does REWRITE MASTERREC after shipping (updates QOH)
- INVALID KEY -> display "PRODUCT NOT FOUND", write pending

VSAM KSDS implemented as dict-based key-value store (load all records,
index by product_id for O(1) lookup).

PERFORM THRU pattern: 1000-INIT-PARA THRU 1000-EXIT-PARA merged into
single function (EXIT paragraph is a no-op).

4 files: INFILE1 (master indexed), INFILE2 (trans),
         OUTFILE1 (shipping), OUTFILE2 (pending)
"""

from __future__ import annotations

import os
import sys

from utils.file_io import SequentialFileWriter

RECORD_WIDTH = 80

# -- VSAM KSDS record: 5-char product_id (unlike 4-char in sequential programs)
_W_PRODUCT_ID = 5
_W_PRODUCT_NAME = 10
_W_QOH = 3
_W_PRICE = 3
_W_MASTER_FILLER = 59   # 80 - 5 - 10 - 3 - 3 = 59

# Transaction record
_W_TRAN_ID = 2
_W_TRANS_PRODID = 5
_W_QTY_REQ = 3
_W_TRANS_FILLER = 70   # 80 - 2 - 5 - 3 = 70


def run_vsamord() -> None:
    """Main entry point -- equivalent to 0000-MAIN-PARA.

    PERFORM 1000-INIT-PARA THRU 1000-EXIT-PARA (merged, EXIT is no-op).
    PERFORM 2000-PROCESS-PARA THRU 2000-EXIT-PARA.
    PERFORM 3000-CLOSE-PARA THRU 3000-EXIT-PARA.

    JCL DD -> env vars:
        INFILE1   -- master product file (VSAM KSDS, indexed)
        INFILE2   -- transaction input file (sequential)
        OUTFILE1  -- shipping orders output
        OUTFILE2  -- pending orders output
    """
    master_path = os.environ.get("INFILE1", "data/master_vsam.dat")
    trans_path = os.environ.get("INFILE2", "data/trans_vsam.dat")
    ship_path = os.environ.get("OUTFILE1", "data/shipping_vsam.dat")
    pend_path = os.environ.get("OUTFILE2", "data/pending_vsam.dat")

    # Validate input files exist
    for path, name in [(master_path, "INFILE1"), (trans_path, "INFILE2")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found: {path}", file=sys.stderr)
            sys.exit(12)

    # 1000-INIT-PARA: Load VSAM KSDS into dict for random key access
    master_index, master_lines = _load_vsam_ksds(master_path)

    # 2000-PROCESS-PARA: read transactions, process each
    with SequentialFileWriter(ship_path) as ship_writer, \
         SequentialFileWriter(pend_path) as pend_writer:

        for trans in _read_transactions(trans_path):
            product_id = trans["product_id"]

            # READ MASTERIN KEY IS PRODUCT-ID
            if product_id not in master_index:
                # INVALID KEY -- product not found
                print(f"PRODUCT NOT FOUND")
                pend_line = _format_pending_line(
                    trans["tran_id"], product_id, trans["qty_req"], "NA",
                )
                pend_writer.write_line(pend_line)
                continue

            line_idx = master_index[product_id]
            master = _parse_master_record(master_lines[line_idx])

            # CRITICAL: VSAMORD uses >= (greater-or-equal)
            if master["qoh"] >= trans["qty_req"]:
                # Ship the order
                bill = master["price"] * trans["qty_req"]

                ship_line = _format_shipping_line(
                    trans["tran_id"], product_id, trans["qty_req"], bill,
                )
                ship_writer.write_line(ship_line)

                # REWRITE MASTERREC: reduce QOH and update in-place
                master["qoh"] -= trans["qty_req"]
                master_lines[line_idx] = _format_master_line(master)
                _rewrite_file(master_path, master_lines)
            else:
                # Not enough stock -- no REWRITE in VSAMORD NS branch
                pend_line = _format_pending_line(
                    trans["tran_id"], product_id, trans["qty_req"], "NS",
                )
                pend_writer.write_line(pend_line)

    # 3000-CLOSE-PARA: files closed by context managers


def _load_vsam_ksds(path: str) -> tuple[dict[str, int], list[str]]:
    """Load VSAM KSDS file into a dict index for random key access.

    Returns:
        master_index: dict mapping product_id -> line index
        master_lines: list of raw lines (for REWRITE)
    """
    master_lines = []
    master_index: dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            padded = line.ljust(RECORD_WIDTH)
            product_id = padded[0:_W_PRODUCT_ID].rstrip()
            idx = len(master_lines)
            master_lines.append(padded)
            master_index[product_id] = idx

    return master_index, master_lines


def _parse_master_record(line: str) -> dict:
    """Parse a VSAM master record line into a dict.

    Layout (80 bytes, all DISPLAY):
        PRODUCT-ID      PIC X(05)
        PRODUCT-NAME    PIC X(10)
        QOH             PIC 9(03)
        PRICE           PIC 9(03)
        FILLER          PIC X(59)
    """
    line = line.ljust(RECORD_WIDTH)
    pos = 0

    product_id = line[pos:pos + _W_PRODUCT_ID].rstrip()
    pos += _W_PRODUCT_ID

    product_name = line[pos:pos + _W_PRODUCT_NAME].rstrip()
    pos += _W_PRODUCT_NAME

    qoh_raw = line[pos:pos + _W_QOH].strip()
    pos += _W_QOH
    qoh = int(qoh_raw) if qoh_raw else 0

    price_raw = line[pos:pos + _W_PRICE].strip()
    pos += _W_PRICE
    price = int(price_raw) if price_raw else 0

    return {
        "product_id": product_id,
        "product_name": product_name,
        "qoh": qoh,
        "price": price,
    }


def _format_master_line(master: dict) -> str:
    """Format a VSAM master record dict back to an 80-char line."""
    parts = [
        master["product_id"].ljust(_W_PRODUCT_ID)[:_W_PRODUCT_ID],
        master["product_name"].ljust(_W_PRODUCT_NAME)[:_W_PRODUCT_NAME],
        str(master["qoh"]).zfill(_W_QOH)[:_W_QOH],
        str(master["price"]).zfill(_W_PRICE)[:_W_PRICE],
    ]
    return "".join(parts).ljust(RECORD_WIDTH)[:RECORD_WIDTH]


def _read_transactions(path: str):
    """Read VSAMORD transaction records.

    Layout (80 bytes, all DISPLAY):
        TRANID      PIC X(02)
        PRODID      PIC X(05)   -- 5 chars (matches VSAM master key)
        QTY-REQ     PIC 9(03)
        FILLER      PIC X(70)
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            line = line.ljust(RECORD_WIDTH)

            tran_id = line[0:_W_TRAN_ID].rstrip()
            product_id = line[_W_TRAN_ID:_W_TRAN_ID + _W_TRANS_PRODID].rstrip()
            qty_raw = line[_W_TRAN_ID + _W_TRANS_PRODID:_W_TRAN_ID + _W_TRANS_PRODID + _W_QTY_REQ].strip()
            qty_req = int(qty_raw) if qty_raw else 0

            yield {
                "tran_id": tran_id,
                "product_id": product_id,
                "qty_req": qty_req,
            }


def _format_pending_line(
    tran_id: str, product_id: str, qty: int, status: str,
) -> str:
    """Format VSAMORD pending record as 80-char line.

    Layout (80 bytes, all DISPLAY):
        TRANID      PIC X(02)
        PROD-ID     PIC X(04)
        QTY-REQ     PIC 9(03)
        BILLSTAT    PIC X(05)
        FILLER      PIC X(66)
    """
    parts = [
        tran_id.ljust(2)[:2],
        product_id.ljust(4)[:4],
        str(qty).zfill(3)[:3],
        status.ljust(5)[:5],
    ]
    return "".join(parts).ljust(RECORD_WIDTH)[:RECORD_WIDTH]


def _format_shipping_line(
    tran_id: str, product_id: str, qty: int, bill: int,
) -> str:
    """Format VSAMORD shipping record as 80-char line.

    Layout (80 bytes, all DISPLAY):
        TRANID      PIC X(02)
        PROID       PIC X(04)
        QTY-REQ     PIC 9(03)
        TOTALBILL   PIC 9(05)
        FILLER      PIC X(66)
    """
    parts = [
        tran_id.ljust(2)[:2],
        product_id.ljust(4)[:4],
        str(qty).zfill(3)[:3],
        str(bill).zfill(5)[:5],
    ]
    return "".join(parts).ljust(RECORD_WIDTH)[:RECORD_WIDTH]


def _rewrite_file(path: str, lines: list[str]) -> None:
    """REWRITE pattern: rewrite entire VSAM file after updating a record."""
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


if __name__ == "__main__":
    run_vsamord()
