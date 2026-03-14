"""ORDERNEW -- New order processing program.

Migrated from: ORDERNEW (PROGRAM-ID. ORDERNEW)

Reads transactions from TRANSFL (INFILE2), scans master file (MASTERFL/INFILE1)
for each transaction.  Ships if QOH > QTY-REQ (strict >), otherwise pends.
Does NOT rewrite master file after shipping.

All fields are DISPLAY format (PIC 9, not COMP-3).
Record layouts match CalcbillMasterRecord format.

PEND-REC BILLSTATUS values:
    "NOT  SUFFICIENT" -- product found, QOH > 0 but QOH <= QTY-REQ
    "NOT   AVAILABLE" -- product not found in master

Writes LOGFILE-REC at end with counters.
RETURN-CODE 20 if no transactions read.

5 files: INFILE1 (master), INFILE2 (trans), OUTFILE1 (pend),
         OUTFILE2 (ship), OUTFILE3 (log)
"""

from __future__ import annotations

import os
import sys

from models.product import CalcbillMasterRecord
from utils.file_io import read_records, SequentialFileWriter

RECORD_WIDTH = 80


def run_ordernew() -> None:
    """Main entry point -- equivalent to 0000-MAIN-PARA.

    JCL DD -> env vars:
        INFILE1   -- master product file (sequential scan per transaction)
        INFILE2   -- transaction input file
        OUTFILE1  -- pending orders output
        OUTFILE2  -- shipping orders output
        OUTFILE3  -- log file output
    """
    master_path = os.environ.get("INFILE1", "data/masterfile.dat")
    trans_path = os.environ.get("INFILE2", "data/transfile.dat")
    pend_path = os.environ.get("OUTFILE1", "data/pendingfile.dat")
    ship_path = os.environ.get("OUTFILE2", "data/shippingfile.dat")
    log_path = os.environ.get("OUTFILE3", "data/logfile.dat")

    # Validate input files exist
    for path, name in [(trans_path, "INFILE2"), (master_path, "INFILE1")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found: {path}", file=sys.stderr)
            sys.exit(12)

    # Counters for log record
    read_master_count = 0
    read_trans_count = 0
    write_shipping_count = 0
    write_pending_count = 0

    # Read transactions
    transactions = list(_read_transactions(trans_path))

    if not transactions:
        # RETURN-CODE 20 if no transactions read
        sys.exit(20)

    with SequentialFileWriter(pend_path) as pend_writer, \
         SequentialFileWriter(ship_path) as ship_writer:

        for trans in transactions:

            # Open master file and scan for matching product
            found = False
            for master in read_records(master_path, CalcbillMasterRecord):

                if master.product_id == trans["product_id"]:
                    found = True

                    # CRITICAL: ORDERNEW uses > (strict greater-than)
                    if master.quantity_on_hand > trans["qty_req"]:
                        # Ship the order
                        read_trans_count += 1
                        bill = master.price * trans["qty_req"]

                        ship_line = _format_shipping_line(
                            trans["tran_id"],
                            trans["product_id"],
                            trans["qty_req"],
                            bill,
                        )
                        ship_writer.write_line(ship_line)
                        write_shipping_count += 1

                        # NOTE: ORDERNEW does NOT rewrite master file
                    else:
                        # Not enough stock
                        read_trans_count += 1
                        pend_line = _format_pending_line(
                            trans["tran_id"],
                            trans["product_id"],
                            trans["qty_req"],
                            "NOT  SUFFICIENT",
                        )
                        pend_writer.write_line(pend_line)
                        write_pending_count += 1

                    break  # Found match, stop scanning master

            # AT END: master scan exhausted without match
            read_master_count += 1
            if not found:
                # Product not found in master
                read_trans_count += 1
                pend_line = _format_pending_line(
                    trans["tran_id"],
                    trans["product_id"],
                    trans["qty_req"],
                    "NOT   AVAILABLE",
                )
                pend_writer.write_line(pend_line)
                write_pending_count += 1

    # Write log record
    log_line = _format_log_line(
        read_master_count, read_trans_count,
        write_shipping_count, write_pending_count,
    )
    with SequentialFileWriter(log_path) as log_writer:
        log_writer.write_line(log_line)


def _read_transactions(path: str):
    """Read ORDERNEW transaction records (TRANSACTION-REC).

    Layout (80 bytes, all DISPLAY):
        TRANSACTIONID       PIC X(02)
        PRODUCTID           PIC X(04)
        QUANTITYREQUIRED    PIC 9(03)
        FILLER              PIC X(71)
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            line = line.ljust(RECORD_WIDTH)

            tran_id = line[0:2].rstrip()
            product_id = line[2:6].rstrip()
            qty_raw = line[6:9].strip()
            qty_req = int(qty_raw) if qty_raw else 0

            yield {
                "tran_id": tran_id,
                "product_id": product_id,
                "qty_req": qty_req,
            }


def _format_pending_line(
    tran_id: str, product_id: str, qty: int, status: str,
) -> str:
    """Format ORDERNEW pending record (PEND-REC) as 80-char line.

    Layout (80 bytes, all DISPLAY):
        TRANSACTIONID       PIC X(02)
        PRODUCTID           PIC X(04)
        QUANTITYREQUIRED    PIC 9(03)
        BILLSTATUS          PIC X(15)
        FILLER              PIC X(56)
    """
    parts = [
        tran_id.ljust(2)[:2],
        product_id.ljust(4)[:4],
        str(qty).zfill(3)[:3],
        status.ljust(15)[:15],
    ]
    return "".join(parts).ljust(RECORD_WIDTH)[:RECORD_WIDTH]


def _format_shipping_line(
    tran_id: str, product_id: str, qty: int, bill: int,
) -> str:
    """Format ORDERNEW shipping record (SHIPPINGREC) as 80-char line.

    Layout (80 bytes, all DISPLAY):
        TRANSACTIONID       PIC X(02)
        PRODUCTID           PIC X(04)
        QUANTITYREQUIRED    PIC 9(03)
        TOTALBILL           PIC 9(05)
        FILLER              PIC X(66)
    """
    parts = [
        tran_id.ljust(2)[:2],
        product_id.ljust(4)[:4],
        str(qty).zfill(3)[:3],
        str(bill).zfill(5)[:5],
    ]
    return "".join(parts).ljust(RECORD_WIDTH)[:RECORD_WIDTH]


def _format_log_line(
    read_master: int, read_trans: int,
    write_ship: int, write_pend: int,
) -> str:
    """Format ORDERNEW log record (LOGFILE-REC) as 80-char line.

    Layout (80 bytes):
        MSGMASTER           PIC X(12)   "MASTER READ"
        READMASTERCOUNT     PIC 9(02)
        SPACEMASTER         PIC X(01)   " "
        MSGTRANS            PIC X(12)   "TRANS  READ"
        READTRANSCOUNT      PIC 9(02)
        SPACETRANS          PIC X(01)   " "
        MSGSHIPPING         PIC X(12)   "SHIP WRITTEN"
        WRITESHIPPINGCOUNT  PIC 9(02)
        SPACESHIP           PIC X(01)   " "
        MSGPENDING          PIC X(12)   "PEND WRITTEN"
        WRITEPENDINGCOUNT   PIC 9(02)
        SPACEPEND           PIC X(01)   " "
        FILLER              PIC X(20)
    """
    parts = [
        "READMASTER: ".ljust(12)[:12],
        str(read_master).zfill(2)[:2],
        " ",
        "READTRANS:  ".ljust(12)[:12],
        str(read_trans).zfill(2)[:2],
        " ",
        "WRITESHIP:  ".ljust(12)[:12],
        str(write_ship).zfill(2)[:2],
        " ",
        "WRITEPEND:  ".ljust(12)[:12],
        str(write_pend).zfill(2)[:2],
        " ",
    ]
    return "".join(parts).ljust(RECORD_WIDTH)[:RECORD_WIDTH]


if __name__ == "__main__":
    run_ordernew()
