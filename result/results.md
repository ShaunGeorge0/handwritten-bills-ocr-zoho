# Model Comparison — Handwritten Bill Extraction

## 1. Overall Summary

| Model | Bills Attempted | Bills Succeeded | Failures | Exact Match Acc (%) | Avg Latency (s) | Cost / Bill (USD) | Cost / 100 Bills (USD) |
|---|---|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite | 15 | 15 | 0 | 13.3 | 21.77 | 0.00104 | 0.1 |
| Nemotron Nano 12B VL (OpenRouter, free) | 15 | 15 | 0 | 0.0 | 5.8 | 0.0 | 0.0 |

## 2. Field-Wise Accuracy

| Model | vendor_name (%) | bill_number (%) | date (%) | currency (%) | total_amount (%) | gst_number (%) | description (%) |
|---|---|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite | 100.0 | 100.0 | 80.0 | 100.0 | 93.3 | 93.3 | 20.0 |
| Nemotron Nano 12B VL (OpenRouter, free) | 80.0 | 86.7 | 40.0 | 100.0 | 60.0 | 93.3 | 13.3 |

## 3. Missing, Hallucinated & Incorrect Fields

| Model | Missing | Hallucinated | Incorrect |
|---|---|---|---|
| Gemini 3.5 Flash-Lite | 0 | 1 | 16 |
| Nemotron Nano 12B VL (OpenRouter, free) | 4 | 3 | 27 |

## 4. Ambiguous vs. Non-Ambiguous Bills

| Model | Ambiguous Exact Match (%) | Ambiguous Bills | Non-Ambiguous Exact Match (%) | Non-Ambiguous Bills |
|---|---|---|---|---|
| Gemini 3.5 Flash-Lite | 25.0 | 4 | 9.1 | 11 |
| Nemotron Nano 12B VL (OpenRouter, free) | 0.0 | 4 | 0.0 | 11 |

## 5. Failure Analysis

### Gemini 3.5 Flash-Lite — 13 bill(s) with at least one field off

| Bill | Field | Status | Ground Truth | Extracted |
|---|---|---|---|---|
| bill_01.jpg | date | incorrect | 2025-11-27 | 2025-11-07 |
| bill_01.jpg | description | incorrect | Books purchase | Book purchase |
| bill_02.jpg | total_amount | incorrect | 1030.0 | 1180.0 |
| bill_04.jpg | description | incorrect | Travel booking | Travel and bus hire charges |
| bill_05.jpg | description | incorrect | Design services | Funeral bouquet purchase |
| bill_06.jpg | gst_number | hallucinated | — | 32AADFB1032A1ZN |
| bill_06.jpg | description | incorrect | General purchase | Musical instrument repair and leather change service |
| bill_07.jpg | description | incorrect | Nursery purchase | Purchase of plants and seeds |
| bill_08.jpg | description | incorrect | General purchase | Registration and membership fees |
| bill_09.jpg | date | incorrect | 2024-11-09 | 2024-04-09 |
| bill_09.jpg | description | incorrect | Gift shop purchase | Stapler pin purchase |
| bill_10.jpg | description | incorrect | Cement purchase | Purchase of M.S wire |
| bill_11.jpg | description | incorrect | Grocery purchase | Food purchase from Rahath Foods |
| bill_12.jpg | description | incorrect | General purchase | Food items purchase bill |
| bill_14.jpg | description | incorrect | Grocery purchase | Furniture purchase |
| bill_15.jpg | date | incorrect | 2025-07-16 | 2024-10-14 |
| bill_15.jpg | description | incorrect | General purchase | Food and snack purchase estimate |

### Nemotron Nano 12B VL (OpenRouter, free) — 15 bill(s) with at least one field off

| Bill | Field | Status | Ground Truth | Extracted |
|---|---|---|---|---|
| bill_01.jpg | date | incorrect | 2025-11-27 | 1978-11-25 |
| bill_01.jpg | description | incorrect | Books purchase | CSSM Book House purchase |
| bill_02.jpg | total_amount | incorrect | 1030.0 | 1480.0 |
| bill_02.jpg | description | incorrect | Flower purchase | Market purchase for 400g of one item and 300g of another item. |
| bill_03.jpg | date | incorrect | 2024-12-02 | 2024-12-18 |
| bill_03.jpg | total_amount | incorrect | 3950.0 | 4880.0 |
| bill_04.jpg | bill_number | missing | 023 | — |
| bill_04.jpg | date | incorrect | 2025-07-27 | 2025-07-25 |
| bill_04.jpg | description | incorrect | Travel booking | Bus hire charges to Thekkady from 25/7/25 |
| bill_05.jpg | vendor_name | incorrect | JO DESIGNS | JO DESIGNS
CANOPY APARTMENT |
| bill_05.jpg | date | missing | 2025-12-13 | — |
| bill_05.jpg | description | incorrect | Design services | Bill for Bouquet Purchase |
| bill_06.jpg | total_amount | incorrect | 1800.0 | 4600.0 |
| bill_06.jpg | description | incorrect | General purchase | Service Book Purchase - Phoenix Bass Drum Supplies Sold by Basil Industries Music Store near Kaloor Bank Kalookul Malapuram from Reserve Banc in Cochin from Ernakulam onto cash paid integers apostrophe Cashidge GPay transaction notification application confirmation RFC comedy Phrase Come This document spell chess Kenny GPS PowerPoint Lux figure? |
| bill_07.jpg | date | hallucinated | — | 2021-01-15 |
| bill_07.jpg | description | incorrect | Nursery purchase | Purchase of plants |
| bill_08.jpg | total_amount | incorrect | 1260.0 | 990.0 |
| bill_08.jpg | description | incorrect | General purchase | Membership Registration and Raja Salari Membership |
| bill_09.jpg | date | incorrect | 2024-11-09 | 2023-09-29 |
| bill_09.jpg | description | incorrect | Gift shop purchase | Office Supplies |
| bill_10.jpg | vendor_name | incorrect | JK CEMENT | J.K. Cement Ltd. |
| bill_10.jpg | total_amount | incorrect | 22.0 | 58.33 |
| bill_10.jpg | gst_number | hallucinated | — | 27AABC1234D1ZEA |
| bill_10.jpg | description | incorrect | Cement purchase | Mende purchase - construction materials from J.K. Cement |
| bill_11.jpg | bill_number | incorrect | 6453 | 21318181000312 |
| bill_12.jpg | description | missing | General purchase | — |
| bill_13.jpg | date | incorrect | 2020-03-09 | 2020-09-03 |
| bill_13.jpg | total_amount | incorrect | 3550.0 | 8755.0 |
| bill_13.jpg | description | incorrect | Gift shop purchase | Gift purchase from Alankar Gifts |
| bill_14.jpg | date | hallucinated | — | 2023-07-05 |
| bill_14.jpg | description | incorrect | Grocery purchase | Steel Wooden Furniture purchase |
| bill_15.jpg | vendor_name | incorrect | Philip | ESTIMATE |
| bill_15.jpg | date | incorrect | 2025-07-16 | 2023-10-01 |
| bill_15.jpg | description | missing | General purchase | — |

## 6. Sample Predictions

### Gemini 3.5 Flash-Lite

| Bill | Vendor (extracted) | Total (extracted) | Date (extracted) | GST # (extracted) | Description (extracted) |
|---|---|---|---|---|---|
| bill_01.jpg | CSSM BOOK HOUSE | 3500.0 | 2025-11-07 |  | Book purchase |
| bill_02.jpg | FLOWER MART | 1180.0 | 2024-09-21 |  | Flower purchase |
| bill_03.jpg | JOSE PRINTERS | 3950.0 | 2024-12-02 |  | Printing services for coupons, cards, and covers |
| bill_04.jpg | PRINCY TRAVELS | 29500.0 | 2025-07-27 |  | Travel and bus hire charges |
| bill_05.jpg | JO DESIGNS | 400.0 | 2025-12-13 |  | Funeral bouquet purchase |

### Nemotron Nano 12B VL (OpenRouter, free)

| Bill | Vendor (extracted) | Total (extracted) | Date (extracted) | GST # (extracted) | Description (extracted) |
|---|---|---|---|---|---|
| bill_01.jpg | CSSM BOOK HOUSE | 3500.0 | 1978-11-25 |  | CSSM Book House purchase |
| bill_02.jpg | FLOWER MART | 1480.0 | 2024-09-21 |  | Market purchase for 400g of one item and 300g of another item. |
| bill_03.jpg | Jose Printers | 4880.0 | 2024-12-18 |  | Printing services |
| bill_04.jpg | PRINCY TRAVELS | 29500.0 | 2025-07-25 |  | Bus hire charges to Thekkady from 25/7/25 |
| bill_05.jpg | JO DESIGNS
CANOPY APARTMENT | 400.0 |  |  | Bill for Bouquet Purchase |

## 7. Suggestions for Model Improvement

**Gemini 3.5 Flash-Lite**

- `description` accuracy is low (20.0%) — check which specific bills fail on this field (Section 5) before assuming it's a model-ceiling issue rather than a prompt or image-quality one.
- 1 hallucinated field(s) — consider strengthening the prompt's "never guess, use empty/0 if not visible" instruction, or adding a couple of few-shot examples that show an explicitly-empty field.

**Nemotron Nano 12B VL (OpenRouter, free)**

- `date` accuracy is low (40.0%) — check which specific bills fail on this field (Section 5) before assuming it's a model-ceiling issue rather than a prompt or image-quality one.
- `total_amount` accuracy is low (60.0%) — check which specific bills fail on this field (Section 5) before assuming it's a model-ceiling issue rather than a prompt or image-quality one.
- `description` accuracy is low (13.3%) — check which specific bills fail on this field (Section 5) before assuming it's a model-ceiling issue rather than a prompt or image-quality one.
- 3 hallucinated field(s) — consider strengthening the prompt's "never guess, use empty/0 if not visible" instruction, or adding a couple of few-shot examples that show an explicitly-empty field.
- 4 missing field(s) — check whether these are genuinely illegible on the source image or a case the model reliably skips regardless of legibility.

