# schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

class LineItem(BaseModel):
    description: str = Field(description="Name or description of the item/service")
    quantity: float = Field(default=1.0, description="Quantity of items purchased")
    unit_price: float = Field(default=0.0, description="Price per individual unit")
    total_amount: float = Field(description="Total price for this line item")

class ReceiptData(BaseModel):
    vendor_name: str = Field(description="Name of the business or vendor issuing the bill")
    bill_number: Optional[str] = Field(default=None, description="Invoice or bill reference number if visible")
    date: str = Field(description="Date of purchase in YYYY-MM-DD format")
    currency: str = Field(default="INR", description="3-letter ISO currency code (e.g., INR, USD)")
    line_items: List[LineItem] = Field(default_factory=list, description="List of itemized goods or services")
    subtotal: Optional[float] = Field(default=None, description="Subtotal before taxes")
    tax_amount: Optional[float] = Field(default=0.0, description="Total tax amount (GST/VAT) if visible")
    total_amount: float = Field(description="Final grand total amount paid or due")
    payment_mode: Optional[str] = Field(default="Cash", description="Payment method used (e.g., Cash, Card, UPI)")