# schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(description="Name or description of the item or service purchased")
    quantity: float = Field(default=1.0, description="Quantity of items purchased")
    unit_price: float = Field(default=0.0, description="Price per individual unit")
    total_amount: float = Field(description="Total price for this line item")


class ReceiptData(BaseModel):
    """Canonical schema used everywhere downstream: evaluator, Zoho client, dashboard.

    --- CHANGE ---------------------------------------------------------------
    Ground truth moved from a numeric `tax_amount` field to a `gst_number`
    field (the vendor's GSTIN — a printed/handwritten identifier, not a
    computed amount) and added `description`, a short one-line summary of
    what the bill is for. Both are now genuinely extracted fields, not
    derived ones — the model has to read/infer them off the bill image, see
    the updated PROMPT_TEXT in extractor.py.
    ----------------------------------------------------------------------------
    """
    vendor_name: str = Field(description="Name of the business or vendor issuing the bill")
    bill_number: Optional[str] = Field(default=None, description="Invoice or bill reference number if visible")
    date: str = Field(description="Date of purchase in YYYY-MM-DD format")
    currency: str = Field(default="INR", description="3-letter ISO currency code (e.g., INR, USD)")
    line_items: List[LineItem] = Field(default_factory=list, description="List of itemized goods or services")
    subtotal: Optional[float] = Field(default=None, description="Subtotal before taxes")
    gst_number: Optional[str] = Field(
        default=None,
        description="Vendor's GST identification number (GSTIN) if printed or handwritten on the bill, else null",
    )
    total_amount: float = Field(description="Final grand total amount paid or due")
    payment_mode: Optional[str] = Field(default="Cash", description="Payment method used (e.g., Cash, Card, UPI)")
    description: Optional[str] = Field(
        default=None,
        description=(
            "Short one-line summary of what the bill is for, e.g. 'Grocery purchase', "
            "'Printing services', 'Restaurant bill' — inferred from vendor name and line items"
        ),
    )


# --- CHANGE -----------------------------------------------------------------
# Gemini's structured-output (response_schema) rejects any Pydantic field with
# a `default` / `default_factory`. GeminiReceiptData is the defaults-free twin
# used only for the Gemini call (see extract_with_gemini in extractor.py);
# its JSON is then loaded into the ReceiptData above for everything
# downstream. Updated in lockstep with ReceiptData: gst_number and
# description added as required string fields ("" when not applicable,
# following the same convention already used for bill_number).
# ---------------------------------------------------------------------------

class GeminiLineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total_amount: float


class GeminiReceiptData(BaseModel):
    vendor_name: str
    bill_number: str  # use "" when not visible — Gemini schema can't do Optional+default
    date: str
    currency: str
    line_items: List[GeminiLineItem]
    subtotal: float
    gst_number: str  # use "" when not visible
    total_amount: float
    payment_mode: str
    description: str  # always producible — inferred, not read off the bill